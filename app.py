"""FootFrame — 체스보드 A4 + OpenCV 2D 측정 + 기존 YOLO.

하나의 파이썬 파일로 통합된 실행판입니다. 각 부분의 강의 기법을 주석으로 표시합니다.
사용자 발 길이/발볼 실측값은 이 코드의 보정 상수로 사용하지 않습니다.
A4 Homography는 종이 평면만 교정하며 발 높이에 의한 시차는 남습니다.
"""
from __future__ import annotations

# 공통 라이브러리 — 강의의 OpenCV 처리 함수들이 같은 모듈을 공유합니다.
import base64
import hashlib
import importlib.util
import html
import json
import os
import threading
import traceback
import urllib.parse
from dataclasses import dataclass, asdict, fields
from email.parser import BytesParser
from email.policy import default
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# ========================================================================
# ===== L05·L06 4점 기하·직선 교점·A4 모서리 정밀화 =====
# ========================================================================

# Automatic, conservative 2D A4 border refinement.
#
# Input: an *already automatically detected* paper quadrilateral (TL,TR,BR,BL),
# plus original BGR image. Optional A4 boundary refinement; only supported edges are used.
# No user clicks, no hardcoded foot sizes or camera parameters, no 3D model.
#
# A4 is a planar known-size object, not a calibration of above-plane feet.


@dataclass
class A4Refinement:
    corners: np.ndarray
    accepted: bool
    reason: str
    residual_before: float
    residual_after: float
    edge_inliers: tuple[int,int,int,int]
    edge_sigma_px: tuple[float,float,float,float]


def order_quad(corners):
    p=np.asarray(corners,dtype=np.float64).reshape(4,2)
    centre=p.mean(axis=0)
    angles=np.arctan2(p[:,1]-centre[1],p[:,0]-centre[0])
    p=p[np.argsort(angles)]
    k=np.argmin(p.sum(axis=1))
    p=np.roll(p,-k,axis=0)
    # Force top-right as second point, not bottom-left.
    if p[1,0]<p[3,0]:p=p[[0,3,2,1]]
    return p


def _image_fields(bgr):
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    gray=cv2.GaussianBlur(lab[:,:,0],(5,5),0)
    return gray,hsv


def _safe_bilinear(a,xy):
    """Vectorized bilinear sample from a single-channel image at xy=(x,y)."""
    h,w=a.shape[:2]
    x=np.clip(xy[:,0],0,w-1.01).astype(np.float32)
    y=np.clip(xy[:,1],0,h-1.01).astype(np.float32)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=x0+1;y1=y0+1
    dx=x-x0;dy=y-y0
    return (a[y0,x0]*(1-dx)*(1-dy)+a[y0,x1]*dx*(1-dy)
            +a[y1,x0]*(1-dx)*dy+a[y1,x1]*dx*dy)


def _line_residuals(gray,hsv,quad,*,search_px=12, samples=145):
    """Extract signed paper-to-background candidate edge samples.

    A skin obstruction is masked using color only to stop the fit following
    the calf silhouette. It does not fill invisible paper or fabricate edges.
    """
    h,w=gray.shape
    center=quad.mean(axis=0)
    all_points=[]
    all_weights=[]
    all_scores=[]
    for j in range(4):
        a,b=quad[j],quad[(j+1)%4]
        tangent=(b-a)/max(1.,np.linalg.norm(b-a))
        n=np.array([-tangent[1],tangent[0]],float)
        if np.dot(n,center-(a+b)/2)<0:n=-n
        leng=float(np.linalg.norm(b-a))
        ts=np.linspace(.06,.94,max(50,min(samples,int(leng/2))),dtype=np.float64)
        mids=a[None,:]+ts[:,None]*(b-a)[None,:]
        offsets=np.arange(-search_px,search_px+1,dtype=np.float64)
        ps=mids[:,None,:]+offsets[None,:,None]*n[None,None,:]
        # Compare both sides of every candidate edge: paper is brighter inside.
        # Use a finite difference that tolerates 1-2px blur and mild shadows.
        outer=ps-4*n[None,None,:]
        inner=ps+4*n[None,None,:]
        inside=(outer[:,:,0]>2)&(outer[:,:,0]<w-3)&(outer[:,:,1]>2)&(outer[:,:,1]<h-3)&(inner[:,:,0]>2)&(inner[:,:,0]<w-3)&(inner[:,:,1]>2)&(inner[:,:,1]<h-3)
        score=_safe_bilinear(gray,inner.reshape(-1,2)).reshape(len(ts),len(offsets))-_safe_bilinear(gray,outer.reshape(-1,2)).reshape(len(ts),len(offsets))
        # Check that inner side has paper-like saturation; keep permissive to
        # handle overexposure and grey paper shadows.
        inside_sample=np.rint(inner).astype(int)
        yi=np.clip(inside_sample[:,:,1],0,h-1);xi=np.clip(inside_sample[:,:,0],0,w-1)
        s=hsv[yi,xi,1].astype(np.float32)
        brightness=_safe_bilinear(gray,inner.reshape(-1,2)).reshape(len(ts),len(offsets))
        score=np.where(inside&(s<95)&(brightness>105),score,-1e6)
        best=np.argmax(score,axis=1)
        strengths=score[np.arange(len(ts)),best]
        valid=strengths>10
        points=ps[np.arange(len(ts)),best][valid]
        strengths=np.clip(strengths[valid],0,100)
        all_points.append(points)
        all_weights.append(strengths)
        all_scores.append(float(np.median(np.clip(np.max(score,axis=1),-35,80))))
    return all_points,all_weights,all_scores


def _robust_line(points,weights,*,seed=91):
    """Local RANSAC + orthogonal total least squares; no sklearn dependency."""
    p=np.asarray(points,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if len(p)<18:return None,None,None
    rng=np.random.default_rng(seed)
    best=None;best_score=-1.
    for _ in range(180):
        i,j=rng.choice(len(p),size=2,replace=False)
        d=p[j]-p[i]; d=d/ max(np.linalg.norm(d),1e-12)
        n=np.array([-d[1],d[0]])
        res=np.abs((p-p[i])@n)
        inliers=res<2.3
        score=np.sum(w[inliers])
        if score>best_score:
            best_score=score;best=inliers
    if best is None or best.sum()<18:return None,None,None
    q=p[best];ww=w[best];ww=np.maximum(1.,ww)
    centroid=np.average(q,axis=0,weights=ww)
    X=q-centroid
    cov=(X*ww[:,None]).T@X/ww.sum()
    _,V=np.linalg.eigh(cov)
    normal=V[:,0]
    sigma=np.sqrt(np.average((X@normal)**2,weights=ww))
    if sigma>2.5:return None,None,None
    return (normal,-float(normal@centroid)),int(best.sum()),float(max(sigma,0.2))


def _intersection(l1,l2):
    A=np.array([l1[0],l2[0]],dtype=float)
    b=-np.array([l1[1],l2[1]],dtype=float)
    if abs(np.linalg.det(A))<0.10:return None
    return np.linalg.solve(A,b)


def _support_evidence(gray,hsv,q):
    points,weights,scores=_line_residuals(gray,hsv,q,search_px=3,samples=120)
    return float(np.mean([np.mean(w) if len(w)>8 else 0. for w in weights])),[len(p) for p in points]


def refine_quad(image,quad,*,search_px=12,max_corner_shift_px=18.) -> A4Refinement:
    """Return original points if evidence is insufficient; never force a fit.

    Any assumed unit is image pixels, so resize before calling if desired.
    Only accept adjustment when both geometry & boundary quality improve.
    """
    p=order_quad(quad)
    def fail(reason,before=0,after=0,n=(0,0,0,0),sig=(0.,0.,0.,0.)):
        return A4Refinement(p.copy(),False,reason,float(before),float(after),tuple(n),tuple(sig))
    im=np.asarray(image)
    if im.ndim!=3 or im.shape[2]!=3 or min(im.shape[:2])<120:return fail('invalid_image')
    h,w=im.shape[:2]
    if (p[:,0].min()<3 or p[:,0].max()>w-4 or p[:,1].min()<3 or p[:,1].max()>h-4):return fail('paper_touches_border')
    gray,hsv=_image_fields(im)
    points,weights,_=_line_residuals(gray,hsv,p,search_px=search_px)
    lines=[];ns=[];sig=[]
    for j,(edge_points,weight) in enumerate(zip(points,weights)):
        line,n,sigma=_robust_line(edge_points,weight,seed=73+j)
        if line is None:return fail('insufficient_visible_edge',n=tuple(len(x) for x in points))
        # Important: the visible line support should cover enough of the edge.
        t=p[(j+1)%4]-p[j]
        f=np.dot(edge_points-p[j],t)/max(1.,np.dot(t,t))
        if np.percentile(f,90)-np.percentile(f,10)<.30:return fail('insufficient_spread')
        lines.append(line);ns.append(n);sig.append(sigma)
    refined=[]
    for j in range(4):
        x=_intersection(lines[(j-1)%4],lines[j])
        if x is None:return fail('nearly_parallel_edges',n=ns,sig=sig)
        refined.append(x)
    refined=np.array(refined,np.float64)
    shifts=np.linalg.norm(refined-p,axis=1)
    if np.any(shifts>max_corner_shift_px):return fail('corner_shift_too_large',n=ns,sig=sig)
    if not cv2.isContourConvex(refined.astype(np.float32)):return fail('nonconvex_quad',n=ns,sig=sig)
    old_area=cv2.contourArea(p.astype(np.float32));new_area=cv2.contourArea(refined.astype(np.float32))
    if not .85<new_area/max(old_area,1)<1.15:return fail('area_changed_too_much',n=ns,sig=sig)
    before,bn=_support_evidence(gray,hsv,p)
    after,an=_support_evidence(gray,hsv,refined)
    if after+0.25<before:return fail('edge_contrast_worse',before,after,ns,sig)
    return A4Refinement(refined,True,'accepted',float(before),float(after),tuple(ns),tuple(sig))


def sensitivity_mm(q, endpoint_pairs, *, sigma_corner_px=2., N=500, seed=49):
    """Paper homography sensitivity ONLY; no coverage claim about foot height.

    Endpoint coordinates are in the image plane and must come from an independent
    automatic detector. This experiment estimates corner-localization sensitivity,
    not a confidence interval for the *true* foot length.
    """
    q=order_quad(q).astype(np.float32)
    ep=np.asarray(endpoint_pairs,dtype=np.float32).reshape(-1,2,2)
    if len(ep)!=1:raise ValueError('Supply exactly one independently detected endpoint pair per sensitivity run')
    dst=np.array([[0,0],[210,0],[210,297],[0,297]],np.float32)
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(N):
        pert=q+rng.normal(0,sigma_corner_px,(4,2)).astype(np.float32)
        H=cv2.getPerspectiveTransform(pert,dst)
        if not np.isfinite(H).all():continue
        for pair in ep:
            proj=cv2.perspectiveTransform(pair.reshape(1,2,2),H).reshape(2,2)
            values.append((float(np.linalg.norm(proj[0]-proj[1])),len(values)%len(ep)))
    # Standard deviation of a known image segment's physical projection.
    lengths=np.array([a for a,b in values])
    if len(lengths)==0:raise ValueError('homography failed')
    return {'projected_median_mm':float(np.median(lengths)),
            'corner_jitter_p05_mm':float(np.percentile(lengths,5)),
            'corner_jitter_p95_mm':float(np.percentile(lengths,95)),
            'sample_count':int(len(lengths)),
            'scope':'Only random A4 corner perturbations; excludes toe/heel localization, lens and height parallax.'}


# ========================================================================
# ===== L02 Canny·직선 후보 / 일반 A4 보조 검출 =====
# ========================================================================

# EXPERIMENTAL fully automatic blank-A4 proposal generator from 2D lines.
#
# No training or photo-specific coordinates. This detector is conservative: only
# high-contrast, geometrically consistent candidates are returned. Prefer the
# legacy plain-A4 detector only as an optional fallback. Never accept proposals
# solely because an orange quad can be drawn around a foot.


@dataclass
class Proposal:
    corners: np.ndarray
    confidence: float
    edges_supported: int
    reason: str


def line_geom(segment):
    p=np.array(segment[:2],dtype=float);q=np.array(segment[2:],dtype=float)
    v=q-p;length=np.linalg.norm(v)
    if length<1:return None
    theta=np.degrees(np.arctan2(v[1],v[0]))%180
    normal=np.array([-v[1],v[0]])/length
    return p,q,theta,normal,float(length)


def signed_edge_support(gray,seg,offset=6):
    geom=line_geom(seg)
    if geom is None:return 0.
    p,q,theta,n,length=geom;h,w=gray.shape
    t=np.linspace(.18,.82,21)
    c=p[None,:]+t[:,None]*(q-p)[None,:]
    pa=np.rint(c+n*offset).astype(int);pb=np.rint(c-n*offset).astype(int)
    ok=(pa[:,0]>=0)&(pa[:,0]<w)&(pa[:,1]>=0)&(pa[:,1]<h)&(pb[:,0]>=0)&(pb[:,0]<w)&(pb[:,1]>=0)&(pb[:,1]<h)
    if ok.sum()<15:return 0.
    a=gray[pa[ok,1],pa[ok,0]].astype(float);b=gray[pb[ok,1],pb[ok,0]].astype(float)
    return float(abs(np.median(a-b)))


def intersect(a,b):
    pa,qa,*_=line_geom(a);pb,qb,*_=line_geom(b)
    va=qa-pa;vb=qb-pb
    A=np.stack((va,-vb),axis=1)
    if abs(np.linalg.det(A))<.06*np.linalg.norm(va)*np.linalg.norm(vb):return None
    t=np.linalg.solve(A,pb-pa)
    return pa+t[0]*va


def proposal_score(gray,hsv,quad,local_std):
    h,w=gray.shape
    region=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(region,np.rint(quad).astype(int),1)
    inner=cv2.erode(region,np.ones((17,17),np.uint8))
    # paper area can be occluded but must contain enough near-white pixels;
    # the highly textured floor alone is not a valid A4.
    if inner.sum()<.06*h*w:return None
    c=gray[inner>0];S=hsv[:,:,1][inner>0]
    bright=float(np.mean((c>145)&(S<83)))
    if bright<.18:return None
    # Wood-grain and other flat surfaces frequently form plausible rectangles.
    # Real blank A4 interior is mostly smooth even with an occluding foot.
    paper_like=(inner>0)&(gray>145)&(hsv[:,:,1]<83)
    if np.count_nonzero(paper_like)<600:return None
    smooth=float(np.mean(local_std[paper_like]<4.))
    if smooth<.67:return None
    center=quad.mean(axis=0);supports=[]
    skin_mask=((hsv[:,:,0]<30)&(hsv[:,:,1]>20)&(hsv[:,:,2]>45))
    for j in range(4):
        a,b=quad[j],quad[(j+1)%4];d=b-a;length=np.linalg.norm(d)
        if length<55:return None
        n=center-(a+b)/2;n=n/max(np.linalg.norm(n),1)
        t=np.linspace(.10,.90,34);c=a[None,:]+t[:,None]*d[None,:]
        inn=np.rint(c+9*n[None,:]).astype(int);out=np.rint(c-9*n[None,:]).astype(int)
        ok=(inn[:,0]>=0)&(inn[:,0]<w)&(inn[:,1]>=0)&(inn[:,1]<h)&(out[:,0]>=0)&(out[:,0]<w)&(out[:,1]>=0)&(out[:,1]<h)
        inn=inn[ok];out=out[ok]
        if len(inn)<18:return None
        diff=gray[inn[:,1],inn[:,0]].astype(float)-gray[out[:,1],out[:,0]].astype(float)
        skin=skin_mask[inn[:,1],inn[:,0]]
        supports.append((float(np.percentile(diff,44)),float(np.mean(diff>11)),float(np.mean(skin))))
    good=sum(1 for contrast,portion,skin in supports if contrast>11 and portion>.45)
    if good<3:return None
    if any(skin>.68 for c,f,skin in supports):return None
    contrast=float(np.sum(np.clip([x[0] for x in supports],-20,80)))
    return (contrast*.15+bright*18+good*6+smooth*14),good


def detect_a4(image,*,max_lines=30):
    if image is None or image.ndim!=3:return None
    im=image
    h,w=im.shape[:2]
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float)
    gray=cv2.GaussianBlur(lab[:,:,0]+1.2*(128-lab[:,:,2]),(7,7),0)
    gray=np.uint8(np.clip(gray,0,255))
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    gl=gray.astype(np.float32)
    local_mean=cv2.blur(gl,(13,13))
    local_std=np.sqrt(np.maximum(0.,cv2.blur(gl*gl,(13,13))-local_mean*local_mean))
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if raw is None:return None
    lines=raw.reshape(-1,4)
    nearH=[];nearV=[]
    for p in lines:
        g=line_geom(p)
        if g is None:continue
        a,b,ang,n,length=g
        if length<.09*min(h,w):continue
        contrast=signed_edge_support(gray,p)
        if contrast<9:continue
        # The sample pictures can be rotated; flexible orientation windows.
        if ang<35 or ang>145:nearH.append((length*contrast,p))
        elif 48<ang<132:nearV.append((length*contrast,p))
    if len(nearH)<2 or len(nearV)<2:return None
    nearH.sort(key=lambda z:z[0],reverse=True)
    nearV.sort(key=lambda z:z[0],reverse=True)
    hs=[e[1] for e in nearH[:max_lines]]
    vs=[e[1] for e in nearV[:max_lines]]
    best=None
    for ia,a in enumerate(hs):
        for ib,b in enumerate(hs):
            if ia==ib:continue
            ay=np.mean(a[1::2]);by=np.mean(b[1::2])
            if ay+min(h,w)*.10>by:continue
            for ic,c in enumerate(vs):
                tl=intersect(a,c);bl=intersect(b,c)
                if tl is None or bl is None:continue
                for id,d in enumerate(vs):
                    if ic==id:continue
                    tr=intersect(a,d);br=intersect(b,d)
                    if tr is None or br is None:continue
                    p=np.array((tl,tr,br,bl),dtype=float)
                    if not np.isfinite(p).all():continue
                    if (p[:,0].min()<w*.015 or p[:,0].max()>w*.985 or
                        p[:,1].min()<h*.015 or p[:,1].max()>h*.985):continue
                    if not cv2.isContourConvex(p.astype(np.float32)):continue
                    area=cv2.contourArea(p.astype(np.float32))/(h*w)
                    if not .085<area<.34:continue
                    lens=np.linalg.norm(p-np.roll(p,-1,axis=0),axis=1)
                    opp=max(lens[0],lens[2])/min(lens[0],lens[2]);opp2=max(lens[1],lens[3])/min(lens[1],lens[3])
                    if opp>1.9 or opp2>1.9:continue
                    mean_horizontal=.5*(lens[0]+lens[2]);mean_vertical=.5*(lens[1]+lens[3])
                    if not .38<mean_horizontal/max(mean_vertical,1)<2.6:continue
                    score=proposal_score(gray,hsv,p,local_std)
                    if score is None:continue
                    val,valid=score
                    if best is None or val>best.confidence:best=Proposal(p,val,valid,'unverified_proposal')
    return best


# ========================================================================
# ===== L02 Thresholding·Contours / 일반 A4 보조 경로 =====
# ========================================================================

# 2D blank A4 detection. Legacy line-segment fallback + independent paper evidence.

class DetectionFailure(ValueError):
    pass


def resize_image(image, longest=950):
    h,w=image.shape[:2]
    if max(h,w)<=longest:return image.copy()
    s=longest/max(h,w)
    return cv2.resize(image, (round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)


def _candidate_contours(image):
    """Independent color/contour hypothesis for white sheets. Never trusts the mask alone."""
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    l=lab[:,:,0]
    h,w=l.shape
    sm=cv2.GaussianBlur(l,(13,13),0)
    # Adaptive local brightness + low saturation, robust to moderately dark pictures.
    local=cv2.GaussianBlur(sm,(0,0),sigmaX=31)
    mask=np.uint8(((sm>local+7)&(hsv[:,:,1]<83)))*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((17,17),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in sorted(cnts,key=cv2.contourArea,reverse=True)[:12]:
        area=cv2.contourArea(c)/(h*w)
        if not .045<area<.55:continue
        hull=cv2.convexHull(c)
        peri=cv2.arcLength(hull,True)
        approx=cv2.approxPolyDP(hull,.03*peri,True)
        if len(approx)==4:
            out.append(approx.reshape(4,2).astype(float))
    return out


def _paper_proof(image, quad):
    h,w=image.shape[:2]
    q=order_quad(quad)
    if not np.isfinite(q).all() or np.min(q)<-1 or q[:,0].max()>w+1 or q[:,1].max()>h+1:
        return None
    area=cv2.contourArea(q.astype(np.float32))/(h*w)
    if not (.055<area<.45):return None
    lens=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
    if min(lens)<.13*min(h,w):return None
    if not cv2.isContourConvex(q.astype(np.float32)):return None
    if max(lens[0]/lens[2],lens[2]/lens[0],lens[1]/lens[3],lens[3]/lens[1])>2.0:return None
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    l=lab[:,:,0]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    m=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(m,np.rint(q).astype(np.int32),1)
    inner=cv2.erode(m,np.ones((21,21),np.uint8))>0
    if inner.sum()<800:return None
    avg=cv2.blur(l,(21,21));std=np.sqrt(np.maximum(0,cv2.blur(l*l,(21,21))-avg*avg))
    smooth=float(np.mean(std[inner]<4))
    # False floor+clothes quadrilaterals, especially 11701, have no mostly smooth A4.
    if smooth<.34:return None
    # Skin and hair should not count as paper.
    paperlike=inner & (std<4) & (hsv[:,:,1]<83)
    fraction=float(np.mean(paperlike[inner]))
    if fraction<.24:return None
    # Actual paper needs a verifiable bright-inside boundary on >= 3 edges.
    center=q.mean(axis=0)
    supports=[]
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        tangent=b-a
        normal=np.array([-tangent[1],tangent[0]],float)
        normal/=max(np.linalg.norm(normal),1)
        if np.dot(normal,center-.5*(a+b))<0:normal=-normal
        pts=a[None,:]+np.linspace(.14,.86,43)[:,None]*tangent[None,:]
        inside=np.rint(pts+8*normal).astype(int)
        outside=np.rint(pts-8*normal).astype(int)
        ok=(inside[:,0]>0)&(inside[:,0]<w)&(inside[:,1]>0)&(inside[:,1]<h)&(outside[:,0]>0)&(outside[:,0]<w)&(outside[:,1]>0)&(outside[:,1]<h)
        if ok.sum()<18: supports.append(0.);continue
        iy,ix=inside[ok,1],inside[ok,0]
        oy,ox=outside[ok,1],outside[ok,0]
        diff=(l[iy,ix]-l[oy,ox])
        supports.append(float(np.mean(diff>8)))
    nstrong=sum(a>.35 for a in supports)
    if nstrong<3:return None
    return float(100*smooth+18*fraction+6*sum(supports)+10*nstrong)


def detect_plain_a4(image):
    """Detect sheet automatically; returns verified A4 corners and diagnostics.

    Experimental LSD proposals are always checked against independent smoothness,
    brightness direction and geometric conditions. We do not fabricate a paper.
    """
    if image is None or image.ndim!=3 or image.shape[2]!=3:raise DetectionFailure('색상 이미지가 필요합니다')
    candidates=[]
    p=detect_a4(image,max_lines=16)
    if p is not None:candidates.append((p.corners,'line_intersections'))
    candidates += [(q,'contour') for q in _candidate_contours(image)]
    valid=[]
    for q,source in candidates:
        score=_paper_proof(image,q)
        if score is not None:valid.append((score,q,source))
    if not valid:raise DetectionFailure('A4의 독립된 네 경계를 확인할 수 없습니다. 발이 종이를 가려도 되지만 최소한 3개 테두리의 증거가 필요합니다.')
    score,q,source=max(valid,key=lambda x:x[0])
    result=refine_quad(image,q)
    if result.accepted and _paper_proof(image,result.corners) is not None:
        q=result.corners
    return order_quad(q), {'score':round(score,2),'method':source,'refined':bool(result.accepted),'refinement':result.reason}


def rectify_plain_a4(image,quad,px_per_mm=2.0):
    """Warp only flat PAPER plane, not above-floor foot surfaces."""
    q=order_quad(quad).astype(np.float32)
    lens=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
    horizontal=float((lens[0]+lens[2])/2)
    vertical=float((lens[1]+lens[3])/2)
    if horizontal>=vertical:
        dimensions=(297.,210.)
    else: dimensions=(210.,297.)
    pw,ph=dimensions
    W,H=int(round(pw*px_per_mm)),int(round(ph*px_per_mm))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32)
    Hmat=cv2.getPerspectiveTransform(q,dst)
    return cv2.warpPerspective(image,Hmat,(W,H)),Hmat,dimensions


# ========================================================================
# ===== L02 Thresholding·Morphology·Contours / 검은 체스보드 칸 검출 =====
# ========================================================================

# L07 — Image Correspondence: local square features + geometric consistency.
#
# The A4 checkerboard has repeating visual features. Detect square-like components
# from L02 threshold/morphology and L03 brightness, remove poor feature matches
# and construct a *provisional* ROI using 2D geometry. The original printed chess
# layout is unknown; DO NOT assign mm to individual squares or claim independent
# paper outer-corner detection from this estimate.


@dataclass
class CheckerEstimate:
    corners: np.ndarray
    features: int
    median_square_px: float
    score: float
    verified_outer_border: bool = False
    method: str = 'checkerboard_bbox_unverified'


def detect_squares(image):
    """L02 findContours/approxPolyDP and L03 HSV brightness segmentation."""
    if image is None or image.ndim != 3:
        return []
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    found = []
    for threshold in (65, 78, 90, 102):
        binary = cv2.inRange(v, 0, threshold)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 70 < area < .02 * h * w:
                continue
            x, y, sw, sh = cv2.boundingRect(contour)
            if not .50 < sw / sh < 2.0 or area / (sw * sh) < .48:
                continue
            hull = cv2.convexHull(contour)
            approx = cv2.approxPolyDP(hull, cv2.arcLength(hull, True) * .08, True)
            if len(approx) != 4:
                continue
            inside = np.zeros((h, w), np.uint8)
            cv2.drawContours(inside, [contour], -1, 255, -1)
            outside = cv2.dilate(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            ring = (outside > 0) & (inside == 0)
            if not np.any(ring):
                continue
            contrast = float(np.median(v[ring]) - np.median(v[inside > 0]))
            if contrast < 42:
                continue
            found.append(dict(center=np.array([x+sw/2, y+sh/2], float),
                              side=float(np.sqrt(area)),
                              corners=approx.reshape(4,2).astype(np.float32),
                              contrast=contrast))
    # Merge detections repeated at multiple brightness thresholds.
    unique = []
    for item in sorted(found, key=lambda s: -s['contrast']):
        if all(np.linalg.norm(item['center'] - prev['center']) > .45 * min(item['side'], prev['side'])
               for prev in unique):
            unique.append(item)
    return unique



def detect_squares_adaptive(image):
    """L02 adaptiveThreshold + Morphology, L03 brightness, L07 outlier checks.

    Fallback when global HSV cutoffs fail because part of the A4 is shaded.
    These are ONLY tentative local black-square features, not known paper
    corner coordinates, nor physical-mm landmarks.  They must pass the
    geometric dispersion tests in estimate_checker_reference() before use.
    """
    if image is None or image.ndim != 3:
        return []
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 51, 10)
    dark = cv2.bitwise_and(dark, dark, mask=np.uint8(hsv[:, :, 1] < 130) * 255)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (90 < area < .02 * H * W):
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if x < 2 or y < 2 or x + bw > W - 2 or y + bh > H - 2:
            continue
        if not (.45 < bw / max(bh, 1) < 2.0 and area / (bw * bh) >= .45):
            continue
        approx = cv2.approxPolyDP(cv2.convexHull(contour),
                                 .07 * cv2.arcLength(cv2.convexHull(contour), True), True)
        if len(approx) != 4:
            continue
        # Compare black interior with immediately surrounding pale paper,
        # instead of using a fixed black global threshold.
        padding = int(.4 * max(bw, bh)) + 2
        x0, y0 = max(0, x-padding), max(0, y-padding)
        x1, y1 = min(W, x+bw+padding), min(H, y+bh+padding)
        patch = gray[y0:y1, x0:x1]
        ring_mask = np.ones(patch.shape, bool)
        ring_mask[y-y0:y-y0+bh, x-x0:x-x0+bw] = False
        outside = patch[ring_mask]
        inside = gray[y:y+bh, x:x+bw]
        if outside.size < 30 or np.mean(outside) < 105:
            continue
        contrast = float(np.mean(outside)-np.mean(inside))
        if contrast < 45:
            continue
        results.append(dict(center=np.array([x+bw/2., y+bh/2.], float),
                            side=float(np.sqrt(area)),
                            corners=approx.reshape(4,2).astype(np.float32),
                            contrast=contrast))
    # Reject duplicates and extreme-scale outliers using only square shape.
    unique = []
    for candidate in sorted(results, key=lambda f: -f['contrast']):
        if all(np.linalg.norm(candidate['center']-prior['center']) >
               .45*min(candidate['side'],prior['side']) for prior in unique):
            unique.append(candidate)
    if len(unique) < 4:
        return []
    sizes=np.array([p['side'] for p in unique])
    median=float(np.median(sizes))
    return [p for p in unique if .55*median <= p['side'] <= 1.60*median]

def estimate_checker_reference(image, features=None):
    """Find support among repeated black squares without known printed layout.

    Do NOT extrapolate a single column of squares to an entire A4: require
    reliable features spanning BOTH dimensions, and label unverified even if
    visually plausible. L07: repeatability, locality, outlier rejection.
    """
    if features is None:
        features = detect_squares(image)
    if len(features) < 4:
        return None
    chosen = []
    for candidate in features:
        side = candidate['side']
        group = [f for f in features
                 if .62 < f['side']/side < 1.65
                 and np.linalg.norm(f['center']-candidate['center']) < 10*side]
        if len(group) > len(chosen):
            chosen = group
    if len(chosen) < 4:
        return None
    med_side = float(np.median([p['side'] for p in chosen]))
    centre = np.median(np.array([p['center'] for p in chosen]), axis=0)
    chosen = [p for p in chosen if np.linalg.norm(p['center']-centre) < 10*med_side]
    if len(chosen) < 4:
        return None
    centres = np.array([p['center'] for p in chosen])
    if np.ptp(centres[:,0]) < 2.4*med_side or np.ptp(centres[:,1]) < 2.3*med_side:
        return None
    corners = np.concatenate([p['corners'] for p in chosen]).astype(np.float32)
    (cx,cy), (rw,rh), angle = cv2.minAreaRect(corners)
    # A single visible strip of black squares is not sufficient to infer
    # both page dimensions. The paper aspect ratio is known (210:297),
    # but the unknown printed pattern aspect ratio must NOT be forced to it.
    aspect = min(rw, rh) / max(rw, rh, 1.)
    if aspect < .36:
        return None
    # Exploratory margins are image-space only: this is not marker calibration.
    box = cv2.boxPoints(((cx,cy),(rw+1.2*med_side,rh+1.2*med_side),angle))
    quad = order_quad(box).astype(np.float32)
    h,w = image.shape[:2]
    if (quad[:,0].min() < 1 or quad[:,1].min()<1 or
            quad[:,0].max() >= w-1 or quad[:,1].max() >= h-1):
        return None
    score = float(len(chosen) + min(np.ptp(centres[:,0]),np.ptp(centres[:,1])) / med_side)
    return CheckerEstimate(quad,len(chosen),med_side,score)


# ========================================================================
# ===== L05 Homography / 검은 칸 정사각형 형태 비교 =====
# ========================================================================

# L02 contours + L05 Planar Homography: check printed-square geometry.
#
# A printed black checker square must remain approximately square when rectified
# onto the A4 plane. Compare the SAME visible squares under the provisional and
# new partial A4 quadrilaterals. This is a conservative *relative* safeguard,
# not a new physical-mm calibration or proof of correct page corners.


@dataclass(frozen=True)
class CheckerShapeEvidence:
    decision: str
    compared_squares: int
    provisional_median_error: float | None
    candidate_median_error: float | None
    median_error_change: float | None

    def json(self):
        return asdict(self)


def _rectification_matrix(quad):
    """L05 getPerspectiveTransform with known A4 aspect ratio, no print pitch."""
    q = order_quad(quad).astype(np.float32)
    sides = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
    if not np.isfinite(q).all() or np.min(sides) < 20:
        return None
    landscape = (sides[0] + sides[2]) >= (sides[1] + sides[3])
    w, h = (594, 420) if landscape else (420, 594)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.getPerspectiveTransform(q, dst)


def _square_error(quad_points):
    """Length equality and adjacent right-angle error for one square."""
    p = order_quad(np.asarray(quad_points, np.float32))
    sides = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(sides, axis=1)
    if not np.isfinite(lengths).all() or np.min(lengths) <= 5:
        return None
    # Standard deviation of log side lengths is scale-independent.
    side_error = float(np.std(np.log(lengths)))
    angle_error = float(np.mean(np.abs(np.sum(sides * np.roll(sides, -1, axis=0),axis=1)
                              / (lengths * np.roll(lengths, -1)))))
    return side_error + 0.4 * angle_error


def compare_checker_shapes(source_squares, provisional, candidate,
                           *, min_squares=4, max_degradation=0.04):
    """Reject only strongly distorted candidate Homographies.

    Lack of squares is inconclusive, NOT a reason to reject a legitimate edge.
    Source contours are identical for both candidate transforms to avoid
    selecting data based on which candidate gives the nicer result.
    """
    if min_squares < 2 or max_degradation <= 0:
        raise ValueError('Invalid checker-square comparison parameters')
    H0 = _rectification_matrix(provisional)
    H1 = _rectification_matrix(candidate)
    if H0 is None or H1 is None:
        return CheckerShapeEvidence('insufficient_geometry', 0, None, None, None)
    q0 = order_quad(provisional).astype(np.float32)
    q1 = order_quad(candidate).astype(np.float32)
    scores0, scores1 = [], []
    for feature in source_squares:
        src = np.asarray(feature.get('corners', []), np.float32)
        center = np.asarray(feature.get('center', [np.nan, np.nan]), np.float64)
        if src.shape != (4, 2) or not np.isfinite(src).all():
            continue
        if not np.isfinite(center).all():
            continue
        # All tested printed squares must lie within BOTH paper proposals.
        # Never use an unrelated black quadrilateral in the background.
        if (cv2.pointPolygonTest(q0, tuple(map(float, center)), False) < 0 or
            cv2.pointPolygonTest(q1, tuple(map(float, center)), False) < 0):
            continue
        t0 = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H0).reshape(4, 2)
        t1 = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H1).reshape(4, 2)
        e0, e1 = _square_error(t0), _square_error(t1)
        if e0 is not None and e1 is not None:
            scores0.append(e0)
            scores1.append(e1)
    n = len(scores0)
    if n < min_squares:
        return CheckerShapeEvidence('insufficient_squares', n, None, None, None)
    before = float(np.median(scores0))
    after = float(np.median(scores1))
    change = after - before
    decision = ('reject_distorted_refit' if change > max_degradation
                else 'no_strong_checker_contradiction')
    return CheckerShapeEvidence(decision, n, round(before, 4),
                                round(after, 4), round(change, 4))


# ========================================================================
# ===== L02 Contours + L07 Outlier Rejection / 칸 크기 반복성 =====
# ========================================================================

# L02 contours + L07 outlier rejection: printed-square consistency audit.
#
# For the ORIGINAL user-provided A4 chessboard, printed-square pitch and margin
# are unknown. Consequently these checks are diagnostic only: they do NOT
# invent an mm size for any square, replace four verified outer corners, or
# change foot length or width based on a person's known dimensions.


@dataclass(frozen=True)
class CheckerQuality:
    detected_squares: int
    supported_squares: int
    median_square_side_px: float | None
    relative_mad: float | None
    status: str
    batch_median_side_px: float | None = None
    batch_scale_deviation: float | None = None

    def json(self):
        return asdict(self)


def audit_checker_squares(rectified) -> CheckerQuality:
    """Estimate the common black-square pixel side after A4 plane homography.

    Partially covered squares are smaller than complete squares. Use the
    largest quartile as a *scale proposal* and keep nearby repeated squares.
    Reject single tiny blobs and never infer a printed physical pitch.
    """
    if rectified is None:
        return CheckerQuality(0, 0, None, None, 'not_available')
    detected = detect_squares(rectified)
    h, w = rectified.shape[:2]
    sides = np.array([float(s['side']) for s in detected
                      if min(h, w) * .065 < s['side'] < min(h, w) * .38], dtype=float)
    if len(sides) < 3:
        return CheckerQuality(len(detected), len(sides), None, None, 'insufficient_squares')
    upper = float(np.percentile(sides, 75))
    consistent = sides[(sides >= .75*upper) & (sides <= 1.22*upper)]
    if len(consistent) < 3:
        return CheckerQuality(len(detected), len(consistent), None, None, 'insufficient_squares')
    center = float(np.median(consistent))
    relative_mad = float(np.median(np.abs(consistent-center)) / center)
    status = 'supported' if relative_mad <= .13 else 'irregular_square_sizes'
    return CheckerQuality(len(detected), len(consistent), round(center, 2),
                          round(relative_mad, 4), status)


def compare_photos(qualities: Sequence[CheckerQuality]):
    """Cross-photo L07 check for one *same printed A4*, reporting only.

    Because the old printed design's millimetre geometry is unavailable,
    the median square image size is NOT a valid calibration target.
    """
    base = [q.median_square_side_px for q in qualities
            if q.status == 'supported' and q.median_square_side_px is not None]
    if len(base) < 3:
        return [q.json() for q in qualities]
    median = float(np.median(base))
    result=[]
    for q in qualities:
        d=q.json()
        d['batch_median_side_px']=round(median,2)
        if q.median_square_side_px is not None and q.status == 'supported':
            deviation=abs(q.median_square_side_px / median - 1.)
            d['batch_scale_deviation']=round(float(deviation),4)
            if deviation>.18:
                d['status']='cross_photo_scale_disagreement'
        result.append(d)
    return result


# ========================================================================
# ===== L03 HSV·Lab 색공간과 L02 Morphology / 발 영역 분할 =====
# ========================================================================

# L03 — Color Spaces and Color Segmentation, L02 — Morphology.
#
# Foot segmentation is still a projected visual silhouette. Skin color varies
# across people and illumination, so any unsupported image must remain unmeasured.


def foot_mask(rect):
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(rect, cv2.COLOR_BGR2LAB)
    skin = ((hsv[:,:,0] < 30) & (hsv[:,:,1] > 24) &
            (hsv[:,:,2] > 33) & (lab[:,:,1] > 122))
    mask = np.uint8(skin) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)
    label = 1 + int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    if stats[label,cv2.CC_STAT_AREA] < .045*mask.size:
        return np.zeros_like(mask)
    mask = np.uint8(labels == label) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))


# ========================================================================
# ===== L02 GaussianBlur·Thresholding·Contours / 흰 종이 외곽 후보 =====
# ========================================================================

# L02/L03/L05/L06: independent 2D A4 border hypothesis from paper color.
#
# An internally printed checkerboard provides an approximate search region only.
# This module searches for the *actual outside paper boundary* using BGR/Lab
# brightness, Thresholding, Morphology, Contours, and edge contrast.  It rejects
# unverifiable polygons instead of calling the black-square bounding box A4.
# There is no guessed square pitch, print margin, 3D correction, or foot-size
# calibration constant in this file.


def _paper_outline_sample(img, pts):
    pts = np.asarray(pts, np.float32)
    return cv2.remap(img.astype(np.float32), pts[..., 0], pts[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)


def _edge_contrast(image, quad):
    """How often is the interior side whiter than the exterior side?"""
    q = order_quad(quad).astype(np.float64)
    gray = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(3,3),0)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,w = gray.shape
    center = q.mean(axis=0)
    supported = []
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        tangent = b-a
        norm = float(np.linalg.norm(tangent))
        if norm < 35:
            return None
        n = np.array([-tangent[1],tangent[0]])/norm
        if np.dot(n,center-(a+b)*0.5) < 0:
            n = -n
        t = np.linspace(.08,.92,55)
        point = a[None,:]+t[:,None]*tangent[None,:]
        ip=point+5*n
        op=point-6*n
        inside=_paper_outline_sample(gray,ip)
        outside=_paper_outline_sample(gray,op)
        saturation=_paper_outline_sample(hsv[:,:,1],ip)
        within=((ip[:,0]>=1)&(ip[:,0]<w-1)&(ip[:,1]>=1)&(ip[:,1]<h-1)&
                (op[:,0]>=1)&(op[:,0]<w-1)&(op[:,1]>=1)&(op[:,1]<h-1))
        contrast = inside-outside
        good = within&(contrast>13)&(inside>110)&(saturation<65)
        supported.append(round(float(np.mean(good)),3))
    return supported


def detect_color_paper_border(image, checker_quad, square_features=None):
    """Return (real A4 corners, diagnostic) or None when uncertain.

    Works only when color/brightness makes the white sheet distinguishable
    from the floor.  A photo with dark shadows, occlusion, or similarly colored
    floor can legitimately fail.  The checker quadrilateral is NEVER treated
    as a complete paper template.
    """
    if image is None or image.ndim != 3:
        return None
    q = order_quad(checker_quad).astype(np.float32)
    H,W = image.shape[:2]
    x,y,w,h=cv2.boundingRect(q)
    pad = int(np.clip(.65*min(w,h),28,240))
    x0,y0=max(0,x-pad),max(0,y-pad)
    x1,y1=min(W,x+w+pad),min(H,y+h+pad)
    if x1-x0<80 or y1-y0<80:
        return None
    marker = np.zeros((H,W),np.uint8)
    cv2.fillConvexPoly(marker,np.rint(q).astype(np.int32),255)
    ring = cv2.dilate(marker,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(111,111)))
    ring[marker!=0] = 0
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blue = image[:,:,0].astype(np.float32)-image[:,:,2].astype(np.float32)
    neutral = (hsv[:,:,1]<36)&(hsv[:,:,2]>105)
    inside = (marker>0)&neutral
    outside = (ring>0)&neutral
    if min(np.count_nonzero(inside),np.count_nonzero(outside))<300:
        return None
    paper_blue = float(np.median(blue[inside]))
    floor_blue = float(np.median(blue[outside]))
    paper_l = float(np.median(gray[inside]))
    floor_l = float(np.median(gray[outside]))
    # Learn color difference from the current photo.  Prefer blue-paper vs
    # warm floor when present; otherwise use relative brightness (L02/L03).
    if paper_blue-floor_blue>=2.7:
        feature = blue
        midpoint=(paper_blue+floor_blue)/2
        method='blue_minus_red'
    elif paper_l-floor_l>=15:
        feature=gray.astype(np.float32)
        midpoint=(paper_l+floor_l)/2
        method='gray_intensity'
    else:
        return None
    foreground = np.uint8((feature>midpoint)&(hsv[:,:,1]<50)&(hsv[:,:,2]>105))*255
    foreground[:y0,:]=0;foreground[y1:,:]=0
    foreground[:,:x0]=0;foreground[:,x1:]=0
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    n,labels,stats,_ = cv2.connectedComponentsWithStats(foreground)
    strongest = None
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<.20*cv2.contourArea(q):
            continue
        component=(labels==i)
        overlap=int(np.count_nonzero(component&(marker>0)))
        if strongest is None or overlap>strongest[0]:
            strongest=(overlap,component)
    if strongest is None:
        return None
    mask=np.uint8(strongest[1])*255
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull=cv2.convexHull(max(cnts,key=cv2.contourArea))
    candidates=[]
    for eps in (.015,.02,.025,.03,.035,.04,.05):
        corners=cv2.approxPolyDP(hull,cv2.arcLength(hull,True)*eps,True)
        if len(corners)==4:
            candidates.append(order_quad(corners.reshape(4,2).astype(np.float32)).astype(np.float32))
    if not candidates:
        return None
    for candidate in candidates:
        if not cv2.isContourConvex(candidate.astype(np.float32)):
            continue
        if (candidate[:,0].min()<x0+2 or candidate[:,0].max()>x1-2 or
            candidate[:,1].min()<y0+2 or candidate[:,1].max()>y1-2):
            continue
        area_ratio = cv2.contourArea(candidate.astype(np.float32))/max(cv2.contourArea(q.astype(np.float32)),1)
        if not .91<area_ratio<1.75:
            continue
        # Outer page must enclose the checker pattern, not a single white patch.
        # Marker corners are hypotheses and may extend slightly beyond paper.
        if square_features:
            covered=sum(cv2.pointPolygonTest(candidate,tuple(map(float,s['center'])),False)>=0
                        for s in square_features)
            coverage = covered/len(square_features)
            if coverage<.94:
                continue
        else:
            coverage=None
        # Require four independently supported paper-to-floor edges.
        support=_edge_contrast(image,candidate)
        if support is None or min(support)<.32:
            continue
        if np.median(support)<.52:
            continue
        return candidate,{
            'method':'l02_l03_contour_verified_outer_border',
            'outer_border_verified':True,
            'color_feature':method,
            'boundary_support':support,
            'checker_inclusion':coverage,
            'area_ratio':round(float(area_ratio),3),
        }
    return None


# ========================================================================
# ===== L02 Canny·HoughLinesP / A4 외곽선 후보 =====
# ========================================================================

# L02 Canny+Hough, L06 line intersections: independent A4 boundary test.
#
# A checker bounding-box is *not* the A4 page outline. This module conservatively
# checks whether long, bright-inside/dark-outside lines support all four sides.
# If evidence is incomplete it returns None, never fabricates calibrated corners.


def detect_outer_border(image, candidate, *, max_offset_px=35):
    """Search real Canny/Hough page-edge lines near a checkerboard hypothesis.

    Returns (corners, diagnostics) only when all 4 independently supported
    boundaries exist. The mm conversion remains A4 210x297, never a guessed
    printed black-square size. Geometry is 2D only.
    """
    q = order_quad(candidate).astype(np.float64)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 115)
    segments = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,
                               minLineLength=max(38, int(min(h,w)*.07)), maxLineGap=16)
    if segments is None:
        return None
    segments = segments.reshape(-1,4)
    center = np.mean(q,axis=0)
    matched=[]
    support=[]
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        delta = b-a
        length = float(np.linalg.norm(delta))
        u=delta/max(length,1.)
        n=np.array([-u[1],u[0]])
        if np.dot(n,center-(a+b)/2)<0:
            n=-n
        picks=[]
        for x1,y1,x2,y2 in segments:
            s=np.array([x1,y1],dtype=float)
            e=np.array([x2,y2],dtype=float)
            direction=e-s
            seg_len=np.linalg.norm(direction)
            if seg_len < max(45,.48*length):
                continue
            line_dir=direction/seg_len
            if abs(float(np.dot(u,line_dir))) < .97:
                continue
            center_seg=.5*(s+e)
            distance=abs(float(np.dot(center_seg-a,n)))
            if distance > max_offset_px:
                continue
            # Independently check the exterior photo pixels (paper vs floor).
            # Hough checker internal edges should not pass this test because
            # their long boundary contrast is not coherent along page outside.
            t=np.linspace(.12,.88,30)
            pts=s[None,:]+t[:,None]*(e-s)[None,:]
            inside=np.rint(pts+7*n).astype(int)
            outside=np.rint(pts-7*n).astype(int)
            valid=(inside[:,0]>=1)&(inside[:,0]<w-1)&(inside[:,1]>=1)&(inside[:,1]<h-1)&(outside[:,0]>=1)&(outside[:,0]<w-1)&(outside[:,1]>=1)&(outside[:,1]<h-1)
            if np.count_nonzero(valid)<18:
                continue
            ip=inside[valid];op=outside[valid]
            contrast=blur[ip[:,1],ip[:,0]].astype(float)-blur[op[:,1],op[:,0]].astype(float)
            fraction=float(np.mean(contrast>7))
            if fraction < .65 or np.median(contrast) < 8:
                continue
            score=fraction*seg_len-.6*distance
            picks.append((score, s, e, fraction))
        if not picks:
            return None
        _,s,e,confidence=max(picks,key=lambda t:t[0])
        matched.append((s,e))
        support.append(confidence)
    intersections=[]
    for prev,cur in zip(matched[-1:]+matched[:-1],matched):
        a,b=prev; c,d=cur
        d1=b-a;d2=d-c
        A=np.column_stack((d1,-d2))
        if abs(np.linalg.det(A))<30:
            return None
        try:
            z=np.linalg.solve(A,c-a)
        except np.linalg.LinAlgError:
            return None
        intersections.append(a+z[0]*d1)
    new=order_quad(intersections).astype(np.float32)
    if not cv2.isContourConvex(new):
        return None
    if (new[:,0].min()<3 or new[:,0].max()>w-4 or
            new[:,1].min()<3 or new[:,1].max()>h-4):
        return None
    shifts=np.linalg.norm(new-q,axis=1)
    if np.max(shifts)>max_offset_px*1.9:
        return None
    a0=cv2.contourArea(q.astype(np.float32));a1=cv2.contourArea(new)
    if not .70<a1/max(a0,1.)<1.35:
        return None
    return new, {'edge_support': [round(x,2) for x in support],
                 'corner_shift_px':round(float(np.median(shifts)),2),
                 'method':'canny_hough_verified_outer_border'}


# ========================================================================
# ===== L07 RANSAC / 끊어진 종이 외곽 직선 적합 =====
# ========================================================================

# L02 Canny/HoughLinesP + L07 two-point RANSAC for A4 outside edges.
#
# A4 is a *plane*, and a collection of long/short nearly-collinear border
# segments can support its 4 sides.  This routine does not use camera pose,
# 3D reconstruction, square pitch, physical foot size, or pretrained networks.
#
# The checker candidate is only a search window. Four outer edges must pass a
# bright-inside/darker-outside test separately before calibration is accepted.


def fit_line_ransac(points, *, trials=96, threshold_px=2.4):
    """L07 slide 54: generate a line from 2 points, count inliers, refit.

    Returns a normal-form line (a,b,c), inliers, or (None, None). A fixed seed
    makes results reproducible for the same photograph.
    """
    pts=np.asarray(points,np.float64).reshape(-1,2)
    if len(pts)<4 or not np.isfinite(pts).all():return None,None
    rng=np.random.default_rng(17)
    best=None;best_mask=None;best_score=-1
    for _ in range(trials):
        i,j=rng.choice(len(pts),size=2,replace=False)
        diff=pts[j]-pts[i]
        length=np.linalg.norm(diff)
        if length<3:continue
        a,b=-diff[1]/length,diff[0]/length
        c=-a*pts[i,0]-b*pts[i,1]
        residual=np.abs(a*pts[:,0]+b*pts[:,1]+c)
        inlier=residual<threshold_px
        score=int(np.count_nonzero(inlier))
        if score>best_score:
            best_score=score;best=np.array([a,b,c]);best_mask=inlier
    if best is None or best_score<4:return None,None
    line=cv2.fitLine(pts[best_mask].astype(np.float32),cv2.DIST_L2,0,0.01,0.01).flatten()
    direction=np.asarray(line[:2],float)
    normal=np.array([-direction[1],direction[0]],float)
    origin=np.asarray(line[2:],float)
    return np.r_[normal,-normal@origin],best_mask


def _sample_gray(gray, pts):
    xy=np.asarray(pts,np.float32)
    return cv2.remap(gray,xy[...,0],xy[...,1],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def _intersect(l1,l2):
    matrix=np.stack((l1[:2],l2[:2]))
    if abs(np.linalg.det(matrix))<.02:return None
    return np.linalg.solve(matrix,-np.array([l1[2],l2[2]]))


def detect_border_ransac(image,checker_quad, *, max_offset_px=50):
    """Propose new A4 quadrilateral if four independent outside edges agree.

    Short line segments are clustered in *normal distance* and confirmed
    against the true white-paper / floor transition (L02/L03).
    """
    q=order_quad(checker_quad).astype(np.float64)
    height,width=image.shape[:2]
    gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(5,5),0)
    edges=cv2.Canny(gray,45,110)
    segments=cv2.HoughLinesP(edges,1,np.pi/360,threshold=26,
                              minLineLength=28,maxLineGap=18)
    if segments is None:return None
    segs=segments.reshape(-1,4)
    center=q.mean(axis=0)
    fitted=[];edge_support=[];shifts=[]
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        vector=b-a;length=np.linalg.norm(vector)
        if length<70:return None
        direction=vector/length
        inward=np.array([-direction[1],direction[0]])
        if inward@(center-(a+b)/2)<0:inward=-inward
        groups={}
        for x1,y1,x2,y2 in segs:
            p=np.array([x1,y1],float);e=np.array([x2,y2],float)
            seg=e-p;seglen=np.linalg.norm(seg)
            if seglen<28 or abs(np.dot(direction,seg/seglen))<.974:continue
            d=float(inward@((p+e)*.5-a))
            if not (-max_offset_px<=d<=8.):continue
            # Exclude lines on a very different portion of the scene, such as
            # tile seams far away from the hypothesized paper sheet.
            t1=float(direction@(p-a)/length);t2=float(direction@(e-a)/length)
            if max(t1,t2)<.06 or min(t1,t2)>.94:continue
            coords=p[None,:]+np.linspace(.12,.88,14)[:,None]*(e-p)[None,:]
            inner=coords+6*inward
            outer=coords-6*inward
            good=(inner[:,0]>1)&(inner[:,0]<width-2)&(inner[:,1]>1)&(inner[:,1]<height-2)&(outer[:,0]>1)&(outer[:,0]<width-2)&(outer[:,1]>1)&(outer[:,1]<height-2)
            if np.count_nonzero(good)<8:continue
            difference=_sample_gray(gray,inner[good])-_sample_gray(gray,outer[good])
            if np.mean(difference>6)<.55 or np.median(difference)<6:continue
            key=int(round(d/5.))
            groups.setdefault(key,[]).append((p,e,seglen))
        if not groups:return None
        choices=[]
        for key,values in groups.items():
            # Require multiple independent short lines or one long one.
            countlen=sum(v[2] for v in values)
            spans=[]
            for p,e,_ in values:
                spans.extend([float(direction@(p-a)/length),float(direction@(e-a)/length)])
            coverage=max(spans)-min(spans)
            if countlen<.31*length or coverage<.37:continue
            pts=np.array([v for p,e,_ in values for v in (p,e)],float)
            # L07 RANSAC uses only observed data, never fabricated points.
            # A single long Hough segment already defines a line from its
            # two observed endpoints; otherwise use the lecture RANSAC fit.
            if len(pts)==2 and countlen>.62*length:
                delta=pts[1]-pts[0]
                tangent=delta/np.linalg.norm(delta)
                normal=np.array([-tangent[1],tangent[0]])
                line=np.r_[normal,-normal@pts[0]]
            else:
                line,inlier=fit_line_ransac(pts)
            if line is None:continue
            normal=line[:2]
            if abs(normal@direction)>.17:continue
            score=countlen/length + .5*coverage - .015*abs(key*5)
            choices.append((score,line,float(key*5),countlen/length))
        if not choices:return None
        _,line,offset,support=max(choices,key=lambda v:v[0])
        fitted.append(line);shifts.append(offset);edge_support.append(round(float(min(support,1.0)),3))
    quad=[]
    for j in range(4):
        point=_intersect(fitted[j-1],fitted[j])
        if point is None:return None
        quad.append(point)
    new=order_quad(np.asarray(quad,np.float32)).astype(np.float32)
    if not cv2.isContourConvex(new):return None
    if (new[:,0].min()<3 or new[:,0].max()>width-3 or
        new[:,1].min()<3 or new[:,1].max()>height-3):return None
    # Avoid passing internal checker edges as outer borders.
    area0=cv2.contourArea(q.astype(np.float32))
    area1=cv2.contourArea(new)
    if not (1.035<area1/max(area0,1.)<1.55):return None
    corner_displacement=np.linalg.norm(new-q,axis=1)
    if np.max(corner_displacement)>max_offset_px*2.0:return None
    contrast=_edge_contrast(image,new)
    if contrast is None or min(contrast)<.25 or np.median(contrast)<.45:return None
    return new, {'method':'l02_l07_canny_hough_ransac_outer_border',
                 'edge_support':edge_support,
                 'edge_contrast':contrast,
                 'corner_shift_px':round(float(np.median(corner_displacement)),2)}


# ========================================================================
# ===== L02 Canny·L03 HSV / 종이와 바닥 경계의 독립 근거 =====
# ========================================================================

# L02/L03/L07: independently check visible outside A4 paper edges.
#
# This audits paper-to-floor brightness and color contrast at each of the four
# outer edges.  A repeating black-square interior is never considered a known
# physical reference. Missing/occluded edges remain explicitly unverified.
#
# This module cannot claim that an estimated border is anatomically accurate,
# correct height-parallax or establish a printed square size in millimeters.


@dataclass(frozen=True)
class EdgeObservation:
    side: str
    status: str
    offset_px: float | None
    support_fraction: float
    candidate_line: list[list[float]] | None
    canny_support_fraction: float = 0.0

    def json(self):
        return asdict(self)


def _paper_edge_audit_sample(channel, points):
    p = np.asarray(points, np.float32)
    # OpenCV remap on uint8 returns uint8. Subtracting two sampled image
    # values would then wrap at zero and invert the meaning of contrast.
    return cv2.remap(channel.astype(np.float32), p[..., 0], p[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def audit_outer_edges(image, provisional_quad, *, max_shift_px=55,
                      min_support=.48, min_canny_support=.34):
    """Find each outer paper edge separately; does not modify the quad.

    Uses only L02/L03 CV primitives: grayscale, HSV, local edge contrast and
    sampling along a line. A candidate must show a broad, consistent bright
    paper *inside* and darker non-black floor *outside*; interior black checker
    squares are excluded by the outside brightness gate.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('BGR image required')
    q = order_quad(np.asarray(provisional_quad, np.float32)).astype(np.float64)
    if not np.isfinite(q).all() or not cv2.isContourConvex(q.astype(np.float32)):
        raise ValueError('valid convex A4 proposal required')
    h, w = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    # L02 Canny verifies an actual edge instead of treating a broad
    # illumination gradient as if it were the physical paper boundary.
    canny = cv2.Canny(gray, 25, 85)
    near_canny = cv2.dilate(canny, np.ones((5, 5), np.uint8))
    center = q.mean(axis=0)
    shift_range = np.arange(-int(max_shift_px), int(max_shift_px)+1, 2, dtype=np.float64)
    ts = np.linspace(.08, .92, 90, dtype=np.float64)
    result = []
    for name, a, b in zip(('top', 'right', 'bottom', 'left'), q, np.roll(q, -1, axis=0)):
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 30:
            result.append(EdgeObservation(name, 'insufficient_edge_evidence', None, 0., None))
            continue
        n = np.array([-edge[1], edge[0]]) / length
        if float(n @ (center - (a + b) / 2)) < 0:
            n = -n
        base = a[None, :] + ts[:, None] * edge[None, :]
        positions = base[:, None, :] + shift_range[None, :, None] * n
        # Compare two sides of a 2D edge, not the already printed chessboard.
        inner = positions + 6*n
        outer = positions - 6*n
        farther = positions - 18*n
        ig = _paper_edge_audit_sample(gray, inner)
        og = _paper_edge_audit_sample(gray, outer)
        iss = _paper_edge_audit_sample(sat, inner)
        oss = _paper_edge_audit_sample(sat, outer)
        far_gray = _paper_edge_audit_sample(gray, farther)
        far_sat = _paper_edge_audit_sample(sat, farther)
        valid = ((inner[..., 0] >= 2) & (inner[..., 0] < w-2) &
                 (inner[..., 1] >= 2) & (inner[..., 1] < h-2) &
                 (outer[..., 0] >= 2) & (outer[..., 0] < w-2) &
                 (outer[..., 1] >= 2) & (outer[..., 1] < h-2) &
                 (farther[..., 0] >= 2) & (farther[..., 0] < w-2) &
                 (farther[..., 1] >= 2) & (farther[..., 1] < h-2))
        diff = ig - og  # signed float: dark-inside/bright-outside stays negative
        # Non-black outside prevents selecting the edge of a printed square.
        good = (valid & (ig > 170) & (iss < 85) & (og > 98) & (oss < 135) &
                (far_gray > 98) & (far_sat < 135) & (diff > 14))
        coverage = good.mean(axis=0)
        # The Canny line must occur at the proposed edge, not just elsewhere
        # in the checkerboard interior. Two independent per-edge checks are
        # required: photometric transition and sharp pixel discontinuity.
        canny_coverage = np.mean(_paper_edge_audit_sample(near_canny, positions) > 0, axis=0)
        score = coverage * 2.0 + np.mean(np.where(good, np.clip(diff, 0, 100), 0.), axis=0) / 90.
        supported = (coverage >= min_support) & (canny_coverage >= min_canny_support)
        if np.any(supported):
            j = int(np.argmax(np.where(supported, score, -1.0)))
        else:
            j = int(np.argmax(score))
        support = float(coverage[j])
        sharpness = float(canny_coverage[j])
        shift = float(shift_range[j])
        if not supported[j]:
            result.append(EdgeObservation(name, 'insufficient_edge_evidence', None,
                                          round(support, 3), None, round(sharpness, 3)))
            continue
        p0 = a + shift*n
        p1 = b + shift*n
        result.append(EdgeObservation(name, 'photometric_edge_candidate', round(shift, 1),
                                      round(support, 3),
                                      [p0.round(2).tolist(), p1.round(2).tolist()],
                                      round(sharpness, 3)))
    return result


def audit_summary(observations):
    valid = [x.side for x in observations if x.status == 'photometric_edge_candidate']
    return {'candidate_edge_count': len(valid), 'candidate_sides': valid,
            'unsupported_sides': [x.side for x in observations if x.side not in valid],
            'all_four_edge_candidates_present': len(valid) == 4,
            'edges': [x.json() for x in observations],
            'is_independently_verified_paper_quad': False,
            'is_physical_accuracy_verified': False,
            'note': '종이 주변의 광도 대비 후보이며 실제 외곽선 4개가 검증되었다는 의미가 아닙니다. 실제 발 치수의 정확도도 보장하지 않습니다.'}


def draw_outer_edge_audit(image, quad, observations):
    output = image.copy()
    cv2.polylines(output, [np.rint(quad).astype(np.int32)], True, (0, 175, 255), 2, cv2.LINE_AA)
    for edge in observations:
        if edge.candidate_line is not None:
            line = np.rint(np.asarray(edge.candidate_line)).astype(np.int32)
            cv2.line(output, tuple(line[0]), tuple(line[1]), (35, 220, 45), 3, cv2.LINE_AA)
            m = np.rint((line[0]+line[1]) * .5).astype(int)
            cv2.putText(output, f'{edge.side} {edge.support_fraction:.2f}',
                        tuple(m), cv2.FONT_HERSHEY_SIMPLEX, .47, (30, 60, 20), 2, cv2.LINE_AA)
    return output


# ========================================================================
# ===== L02 Canny·L07 RANSAC / 검증된 변만 부분 보정 =====
# ========================================================================

# Partial A4 boundary refinement using only L02/L03/L07 OpenCV primitives.
#
# Each of four edges is inspected separately using Canny and signed brightness
# contrast across the *paper-to-floor* boundary. A robust line may replace an
# individual provisional edge only when the measured edge points are plentiful,
# span most of that side, and lie on/OUTSIDE the checkerboard proposal.
#
# No printed checker pitch, known human foot dimension, hand-picked corner, or
# 3D model is used. An unseen edge remains unverified and unmodified. A partial
# refinement is NEVER equivalent to four independently verified A4 corners.


def _paper_line_refinement_sample(channel, xy):
    p=np.asarray(xy, np.float32)
    return cv2.remap(channel.astype(np.float32),p[...,0],p[...,1],cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0)


@dataclass(frozen=True)
class SideFit:
    name: str
    fit_used: bool
    inlier_count: int
    point_count: int
    along_side_fraction: float
    outward_at_start_px: float | None
    outward_at_end_px: float | None
    fitted_segment: list | None
    reason: str

    def json(self):
        return asdict(self)


def _refine_once(image, provisional, *, max_outward_px=55,
                 min_inliers=55, min_side_span=0.55, allowed_sides=None):
    """Return (quadrilateral, diagnostics), or unchanged if no credible edge.

    Orientation-preserving: top/right/bottom/left are according to ordered image
    corners. Perturbations are outward from a checkerboard-bbox polygon only.
    """
    q=order_quad(np.asarray(provisional,np.float32)).astype(np.float64)
    H,W=image.shape[:2]
    if not cv2.isContourConvex(q.astype(np.float32)):
        raise ValueError('A convex quadrilateral is required')
    gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(3,3),0)
    sat=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1]
    canny=cv2.Canny(gray,30,90)
    near=cv2.dilate(canny,np.ones((3,3),np.uint8))
    center=np.mean(q,axis=0)
    result=[]; lines=[]
    names=('top','right','bottom','left')
    for name,a,b in zip(names,q,np.roll(q,-1,axis=0)):
        tangent=b-a
        L=float(np.linalg.norm(tangent))
        if L<65:
            result.append(SideFit(name,False,0,0,0,None,None,None,'side_too_short'))
            lines.append((a,b))
            continue
        inward=np.array([-tangent[1],tangent[0]])/L
        if inward@(center-(a+b)*0.5)<0:
            inward=-inward
        ts=np.linspace(.035,.965,170)
        offsets=np.arange(-float(max_outward_px),19.001,1.0)
        base=a[None,:]+ts[:,None]*tangent
        pts=base[:,None,:]+offsets[None,:,None]*inward
        inside=pts+7*inward
        outside=pts-7*inward
        farther=pts-14*inward
        ig=_paper_line_refinement_sample(gray,inside); og=_paper_line_refinement_sample(gray,outside)
        ins=_paper_line_refinement_sample(sat,inside); oss=_paper_line_refinement_sample(sat,outside)
        fg=_paper_line_refinement_sample(gray,farther)
        valid=((inside[...,0]>2)&(inside[...,0]<W-2)&(inside[...,1]>2)&(inside[...,1]<H-2)&
               (outside[...,0]>2)&(outside[...,0]<W-2)&(outside[...,1]>2)&(outside[...,1]<H-2)&
               (farther[...,0]>2)&(farther[...,0]<W-2)&(farther[...,1]>2)&(farther[...,1]<H-2))
        difference=ig-og
        actual_canny=_paper_line_refinement_sample(near,pts)>0
        good=(valid & (ig>158)&(ins<90)&(og>97)&(oss<135)&
              (fg>93)&(difference>11)&actual_canny)
        score=np.where(good,np.minimum(difference,110),0.)
        indices=np.argmax(score,axis=1)
        found=score[np.arange(len(ts)),indices]>11
        t=ts[found]; shifts=offsets[indices[found]]
        if len(t)<min_inliers:
            result.append(SideFit(name,False,0,len(t),0,None,None,None,'insufficient_edge_points'))
            lines.append((a,b))
            continue
        # L07 two-point RANSAC fit to edge offsets. Reproducible per image.
        rng=np.random.default_rng(17)
        best=None; best_mask=None; best_count=0
        for _ in range(170):
            i,j=rng.choice(len(t),2,replace=False)
            if abs(t[i]-t[j])<.18:
                continue
            slope=(shifts[i]-shifts[j])/(t[i]-t[j]); intercept=shifts[i]-slope*t[i]
            if abs(slope)>44:
                continue
            inliers=np.abs(shifts-(intercept+slope*t))<3.3
            count=int(inliers.sum())
            if count>best_count:
                best=(slope,intercept);best_mask=inliers;best_count=count
        if best is None or best_count<min_inliers:
            result.append(SideFit(name,False,best_count,len(t),0,None,None,None,'no_robust_line'))
            lines.append((a,b))
            continue
        xy=np.c_[t[best_mask],np.ones(best_count)]
        slope,intercept=np.linalg.lstsq(xy,shifts[best_mask],rcond=None)[0]
        inliers=np.abs(shifts-(intercept+slope*t))<3.3
        span=float(np.ptp(t[inliers])) if np.any(inliers) else 0.
        shift0=float(intercept);shift1=float(intercept+slope)
        support=int(inliers.sum())
        # A black-square bounding box is expected to sit inside the page.
        # A line entirely INWARD would likely be a black-square grid edge.
        # Reject excessive skew and large extrapolation near the corners.
        if (support<min_inliers or span<min_side_span or
            max(shift0,shift1)>16.5 or
            min(shift0,shift1)<-max_outward_px-5 or abs(slope)>44):
            result.append(SideFit(name,False,support,len(t),round(span,3),
                                  round(-shift0,1),round(-shift1,1),None,
                                  'insufficient_span_or_outside_geometry'))
            lines.append((a,b))
            continue
        if allowed_sides is not None and name not in allowed_sides:
            result.append(SideFit(name,False,support,len(t),round(span,3),
                                  round(-shift0,1),round(-shift1,1),None,
                                  'not_stable_across_exposures'))
            lines.append((a,b))
            continue
        p0=a+shift0*inward;p1=b+shift1*inward
        result.append(SideFit(name,True,support,len(t),round(span,3),
                              round(-shift0,1),round(-shift1,1),
                              [p0.round(1).tolist(),p1.round(1).tolist()],
                              'observed_canny_and_brightness'))
        lines.append((p0,p1))
    # Project vertices to intersections of neighboring fitted and original lines.
    # Require two supported sides, except when one side alone spans most of its
    # length and lies well outside the provisional box. Never mark as verified.
    used=[x.name for x in result if x.fit_used]
    one_very_strong = (len(used) == 1 and any(
        x.fit_used and x.inlier_count >= 110 and x.along_side_fraction >= .83 and
        x.outward_at_start_px is not None and
        (x.outward_at_start_px + x.outward_at_end_px)/2 >= 6.0
        for x in result))
    if len(used)<2 and not one_very_strong:
        return q.astype(np.float32),{'applied':False,'sides':used,
                                     'reason':'fewer_than_two_independent_edges',
                                     'edge_fits':[x.json() for x in result]}
    normals=[]
    for p0,p1 in lines:
        v=np.asarray(p1)-np.asarray(p0)
        L=np.linalg.norm(v)
        n=np.array([-v[1],v[0]])/max(L,1)
        normals.append(np.r_[n,-n@p0])
    corners=[]
    for i in range(4):
        l1=normals[i-1];l2=normals[i]
        mat=np.array([l1[:2],l2[:2]])
        if abs(np.linalg.det(mat))<.11:
            return q.astype(np.float32),{'applied':False,'sides':used,'reason':'near_parallel_edges',
                                         'edge_fits':[x.json() for x in result]}
        corners.append(np.linalg.solve(mat,-np.array([l1[2],l2[2]])))
    candidate=order_quad(np.array(corners,np.float32)).astype(np.float32)
    original_area=cv2.contourArea(q.astype(np.float32))
    area_ratio=cv2.contourArea(candidate)/max(original_area,1.)
    displacements=np.linalg.norm(candidate-q,axis=1)
    if (not cv2.isContourConvex(candidate) or not np.isfinite(candidate).all() or
        np.min(candidate[:,0])<2 or np.max(candidate[:,0])>W-3 or
        np.min(candidate[:,1])<2 or np.max(candidate[:,1])>H-3 or
        not .96<=area_ratio<=1.43 or np.max(displacements)>75):
        return q.astype(np.float32),{'applied':False,'sides':used,
                                     'reason':'invalid_new_quadrilateral',
                                     'edge_fits':[x.json() for x in result]}
    return candidate,{'applied':True,'sides':used,
                      'reason':'partial_edge_refinement_not_four_corners_verified',
                      'area_ratio':round(float(area_ratio),3),
                      'max_corner_shift_px':round(float(np.max(displacements)),2),
                      'edge_fits':[x.json() for x in result]}



def refine_visible_outer_edges(image, provisional, *, max_outward_px=55,
                               min_inliers=55, min_side_span=0.55,
                               exposure_deltas=(-12, 12),
                               max_line_change_px=5.0):
    """L02 intensity-perturbation repeatability gate for L07 RANSAC lines.

    Run the same paper-edge estimator on a slightly darkened and brightened
    version of the exact same photograph. Only sides whose fitted segments
    agree in BOTH perturbations can be used. This cannot turn partial evidence
    into a full independent A4 boundary verification. It is also not a test
    against manually labelled true page corners or known foot dimensions.

    A piece of floor grout can be stable as well, so this gate is necessary
    additional evidence for partial refits, NOT a certificate of correctness.
    """
    params=dict(max_outward_px=max_outward_px,
                min_inliers=min_inliers,min_side_span=min_side_span)
    baseline, info=_refine_once(image, provisional, **params)
    if not info['applied']:
        info['exposure_stability']='not_tested_no_partial_fit'
        return baseline, info
    stable=set(info['sides'])
    original={fit['name']:np.asarray(fit['fitted_segment'],np.float64)
              for fit in info['edge_fits'] if fit['fit_used']}
    failures={}
    for delta in exposure_deltas:
        shifted=np.clip(image.astype(np.int16)+int(delta),0,255).astype(np.uint8)
        _, alternative=_refine_once(shifted, provisional, **params)
        alt={fit['name']:np.asarray(fit['fitted_segment'],np.float64)
             for fit in alternative['edge_fits'] if fit['fit_used']}
        for name in tuple(stable):
            if name not in alt:
                stable.remove(name)
                failures[name]=f'not_detected_at_{delta:+d}'
            else:
                drift=float(np.max(np.linalg.norm(original[name]-alt[name],axis=1)))
                if drift>max_line_change_px:
                    stable.remove(name)
                    failures[name]=f'line_drift_{drift:.1f}px_at_{delta:+d}'
    if stable == set(info['sides']):
        info['exposure_stability']='all_refitted_sides_stable'
        info['stable_sides']=sorted(stable)
        info['unstable_sides']={}
        return baseline, info
    original_quad=order_quad(np.asarray(provisional,np.float32)).astype(np.float32)
    rerun, updated=_refine_once(image,original_quad,
                                allowed_sides=stable,**params)
    updated['exposure_stability']='unstable_sides_removed'
    updated['stable_sides']=sorted(stable)
    updated['unstable_sides']=failures
    return rerun, updated


def draw_border_refinement(img,old_quad,new_quad,quality):
    out=img.copy()
    cv2.polylines(out,[np.rint(old_quad).astype(np.int32)],True,(0,150,255),2,cv2.LINE_AA)
    for fit in quality['edge_fits']:
        if fit.get('fit_used') and fit.get('fitted_segment') is not None:
            xy=np.rint(fit['fitted_segment']).astype(int)
            cv2.line(out,tuple(xy[0]),tuple(xy[1]),(0,220,40),3,cv2.LINE_AA)
    if quality['applied']:
        cv2.polylines(out,[np.rint(new_quad).astype(np.int32)],True,(255,85,30),2,cv2.LINE_AA)
    unstable=quality.get('unstable_sides') or {}
    if unstable:
        text='EXPOSURE UNSTABLE: '+','.join(sorted(unstable))
        cv2.putText(out,text,(14,max(20, out.shape[0]-14)),
                    cv2.FONT_HERSHEY_SIMPLEX,.53,(10,10,210),2,cv2.LINE_AA)
    return out


# ========================================================================
# ===== L05·L06 Homography·warpPerspective / 체스보드 기준 A4 보정 =====
# ========================================================================

# L05 Planar Homography, L06 Homography Estimation (2D plane only).



def _recover_checker_from_exposure(image, adaptive_features):
    """L02/L03 threshold recovery verified by independent local square positions.

    A small positive exposure perturbation may restore dark printed squares
    clipped by fixed HSV cutoffs. Only accept a geometry when at least four
    corners' source *square centers* agree with the original-image adaptive
    detector. No known physical size or foot measurement enters this search.
    This remains an UNVERIFIED checkerboard bounding box.
    """
    if len(adaptive_features) < 4:
        return None
    raised = np.clip(image.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    recovered = detect_squares(raised)
    reference = estimate_checker_reference(raised, features=recovered)
    if reference is None or len(recovered) < 4:
        return None
    # Corroborate a recovered repeated pattern with the original image.
    matched = []
    seen = set()
    for feature in recovered:
        for i, original in enumerate(adaptive_features):
            if i in seen:
                continue
            tolerance = 0.35 * max(float(feature['side']), float(original['side']))
            if np.linalg.norm(feature['center'] - original['center']) <= tolerance:
                seen.add(i)
                matched.append(feature)
                break
    if len(matched) < 4:
        return None
    # A floor seam can introduce extra dark quadrilaterals after brightening.
    # Never use non-correlated squares to expand the provisional A4 bounding box.
    verified_reference = estimate_checker_reference(raised, features=matched)
    if verified_reference is None:
        return None
    return verified_reference, matched


def detect_paper(image, *, _allow_exposure_fallback=True):
    """Prioritize checkerboard geometry; retain old A4 contour fallback.

    Checker bounding-box corners are NOT independently verified page corners.
    The fallback is preserved for unprinted A4 used by the older 8-photo set.
    """
    detected = detect_squares(image)
    estimated = estimate_checker_reference(image, features=detected)
    fallback = False
    if estimated is None:
        # Recover strong printed squares from a mildly dim exposure only when
        # the original photo independently corroborates their centers.
        adaptive_features = detect_squares_adaptive(image)
        recovery = _recover_checker_from_exposure(image, adaptive_features)
        if recovery is not None:
            estimated, detected = recovery
            fallback = 'exposure_'
        else:
            # L02 adaptiveThreshold remains a provisional fallback. Never
            # call its black-square bounds independently verified paper edges.
            adaptive = estimate_checker_reference(image, features=adaptive_features)
            if adaptive is not None:
                detected = adaptive_features
                estimated = adaptive
                fallback = 'adaptive_'
    if estimated is not None:
        # Two independent checks using only taught 2D CV methods.  Neither
        # changes the reference size without photometric evidence of page edge.
        verified = detect_color_paper_border(image, estimated.corners, detected)
        if verified is None:
            verified = detect_outer_border(image, estimated.corners)
        if verified is None:
            # L02 Canny/Hough: join separately detected short edge fragments.
            # L07 slide 54: two-point RANSAC rejects unrelated floor seams.
            # This never assumes printed checker-square size or 3D anatomy.
            verified = detect_border_ransac(image, estimated.corners)
        if verified is not None:
            corners, evidence = verified
            # Outlying square features should not force a rectangle onto the
            # floor. Require the verified outer sheet to include most of the
            # independently detected black square centers.
            inside = sum(cv2.pointPolygonTest(corners, tuple(map(float, f['center'])), False) >= 0
                         for f in detected)
            if detected and inside / len(detected) < .9:
                verified = None
        if verified is not None:
            corners, evidence = verified
            return corners, {
                'score': round(estimated.score, 2),
                'method': evidence['method'],
                'refined': True,
                'outer_border_verified': True,
                'detected_squares': estimated.features,
                'edge_support': evidence.get('edge_support', evidence.get('boundary_support', [])),
            }
        if _allow_exposure_fallback:
            rescue = _recover_contour_from_exposure(image, estimated.corners, detected)
            if rescue is not None:
                corners, meta, delta = rescue
                return corners, {
                    'score': round(estimated.score,2),
                    'method': 'l02_exposure_confirmed_contour_outer_border',
                    'refined': True,
                    'outer_border_verified': True,
                    'detected_squares': estimated.features,
                    'edge_support': meta.get('boundary_support'),
                    'exposure_delta': delta,
                    'note': 'Heuristic 2D contour verified against the original Canny edges; not physical-foot calibration.',
                }
        # Use L02 Canny, L03 signed photometric differences, L07 RANSAC
        # to correct only independently observed paper/floor boundary segments.
        # Partially observed sides are NOT a verified A4 plane calibration.
        proposed, partial = refine_visible_outer_edges(image, estimated.corners)
        if partial.get('applied'):
            shape_evidence = compare_checker_shapes(detected,
                                                   estimated.corners, proposed)
            partial['checker_shape_guard'] = shape_evidence.json()
            if shape_evidence.decision == 'reject_distorted_refit':
                # A bright stable floor seam is not necessarily the paper edge.
                # If the correction deforms printed square geometry, preserve
                # the provisional board and expose a rejection diagnostic.
                proposed = estimated.corners.copy()
                partial['applied'] = False
                partial['reason'] = 'checker_square_geometry_degraded'
                partial['sides'] = []
        return proposed, {
            'score': round(estimated.score,2),
            'method': ('l02_l07_partial_outer_edge_unverified' if partial['applied']
                       else (fallback + estimated.method if fallback else estimated.method)),
            'refined': bool(partial['applied']),
            'outer_border_verified': False,
            'detected_squares': estimated.features,
            'partial_border_diagnostics': partial,
            'checkerboard_provisional_corners': estimated.corners.round(2).tolist(),
        }
    # Production requires the printed A4 checkerboard. The older plain-A4
    # detector remains available only as an explicit legacy regression option.
    if os.getenv('FOOTFRAME_ALLOW_PLAIN_A4', '0') != '1':
        raise DetectionFailure('체스보드 A4 검은 칸을 충분히 검출하지 못했습니다. 체스보드 A4의 전체 테두리를 촬영해 주세요.')
    quad, info = detect_plain_a4(image)
    info = dict(info)
    info['method'] = 'legacy_plain_a4_' + info['method']
    info['outer_border_verified'] = True  # 2D edge evidence, not ground truth calibration
    info['detected_squares'] = 0
    return quad, info


def _recover_contour_from_exposure(image, proposal, square_features):
    """L02 intensity transforms, Contours and independent Canny validation.

    Input-image brightness must not change the A4 outline. When the normal
    detection is unverified, search nearby exposures for a *color contour*,
    not a Hough/RANSAC black-square coincidence. Validate the candidate on
    the original image; otherwise keep the original provisional estimate.
    """
    q0=order_quad(proposal).astype(np.float32)
    baseline_area=max(cv2.contourArea(q0),1)
    longest=float(max(np.linalg.norm(q0-np.roll(q0,-1,axis=0),axis=1)))
    for delta in (-12,12):
        shifted=np.clip(image.astype(np.int16)+delta,0,255).astype(np.uint8)
        try:
            quad,meta=detect_paper(shifted,_allow_exposure_fallback=False)
        except (DetectionFailure,ValueError,cv2.error):
            continue
        # Repeated checkerboard corners must not be promoted merely because a
        # different exposure generated a geometrically plausible bounding box.
        if meta.get('method') != 'l02_l03_contour_verified_outer_border':
            continue
        quad=order_quad(quad).astype(np.float32)
        area=cv2.contourArea(quad)/baseline_area
        if not .70 < area < 1.40:
            continue
        if np.max(np.linalg.norm(quad-q0,axis=1))>.17*longest:
            continue
        if square_features:
            enclosed=sum(cv2.pointPolygonTest(quad,tuple(map(float,f['center'])),False)>=0
                         for f in square_features)
            if enclosed/len(square_features)<.9:
                continue
        # Original, unmodified pixels must independently support three or
        # more true paper-to-floor contrasts and Canny edge positions.
        evidence=audit_outer_edges(image,quad,max_shift_px=12,
                                   min_support=.35,min_canny_support=.20)
        if sum(e.status=='photometric_edge_candidate' for e in evidence)<3:
            continue
        return quad,meta,delta
    return None


def validate_a4_quadrilateral(quad, image_shape):
    """Reject geometrically degenerate A4 proposals before L05 Homography.

    L02 Contours + L05 polygon geometry: four ordered, distinct convex corners
    must enclose a genuine 2D area. This does NOT prove that the detected
    points are the physical A4 corners; it only rejects impossible transforms.
    No assumed checker-square pitch or known human foot size is used.
    """
    points=np.asarray(quad,dtype=np.float32)
    if points.shape!=(4,2) or not np.isfinite(points).all():
        raise DetectionFailure('A4 모서리 좌표가 유효하지 않습니다.')
    q=order_quad(points).astype(np.float32)
    height,width=image_shape[:2]
    edges=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
    x,y,bw,bh=cv2.boundingRect(q)
    area=float(cv2.contourArea(q))
    # A very narrow ribbon or self-intersection is an unstable plane warp.
    # Ratios below are unitless geometric validity gates, not mm corrections.
    if (height<=1 or width<=1 or not cv2.isContourConvex(q) or
        min(bw,bh)<10 or float(edges.min())<5 or
        area/max(float(bw)*bh,1.0)<0.20 or
        area < 100):
        raise DetectionFailure('A4 사각형이 찌그러져 원근 보정을 신뢰할 수 없습니다.')
    return q


def rectify(image,quad,px_per_mm=2.):
    """cv.getPerspectiveTransform and cv.warpPerspective — L05 pages 8–11."""
    quad=validate_a4_quadrilateral(quad,image.shape)
    sides=np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1)
    horizontal=float((sides[0]+sides[2])/2)
    vertical=float((sides[1]+sides[3])/2)
    paper_mm=(297.,210.) if horizontal>=vertical else (210.,297.)
    width=int(round(paper_mm[0]*px_per_mm))
    height=int(round(paper_mm[1]*px_per_mm))
    target=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    transform=cv2.getPerspectiveTransform(quad,target)
    rect=cv2.warpPerspective(image,transform,(width,height))
    return rect,transform,paper_mm


# ========================================================================
# ===== L05 이미지 좌표 방향 / 상하 회전 안정화 =====
# ========================================================================

# L02 2D image rotation: canonicalize the foot mask on the A4 plane.
#
# A smartphone may produce the same photo rotated by 180 degrees.  The previous
# row-dependent toe/heel measurements changed even though the physical scene was
# identical.  This module only flips when the observed occupied image border
# supports a clear orientation. Ambiguous photos keep the original orientation.
# No knowledge of physical foot size or 3D pose is used.


def normalize_measurement_orientation(rectified, mask, kind):
    if kind not in ('top', 'side'):
        raise ValueError('kind must be top or side')
    if rectified is None or mask is None or rectified.shape[:2] != mask.shape:
        raise ValueError('matching rectified image and mask are required')
    h, _ = mask.shape
    binary = mask > 0
    if kind == 'top':
        # In the provided capture layout, toes terminate near the top of the
        # A4 and the leg connects to the lower paper border. A strong inverse
        # border occupancy is evidence of a 180-degree photo rotation.
        n = max(3, int(.18 * h))
        invert = float(binary[:n].mean() - binary[-n:].mean()) > .06
    else:
        # Side views show the visible foot largely on the top half of the
        # rectified paper with the bottom paper margin empty. Avoid flipping
        # when both ends have similar occupancy (insufficient evidence).
        n = max(3, int(.35 * h))
        invert = float(binary[-n:].mean() - binary[:n].mean()) > .07
    if invert:
        return (cv2.rotate(rectified, cv2.ROTATE_180),
                cv2.rotate(mask, cv2.ROTATE_180), True)
    return rectified, mask, False


# ========================================================================
# ===== L02 Morphology Opening / 측정 마스크 잡음 제거 =====
# ========================================================================

# L02 Morphology: remove isolated thin skin-mask traces before measuring.
#
# A small 3x3 opening removes one-pixel bridges and scattered camera/color
# artifacts. This is an image-space operation; there are no assumed foot sizes,
# learned parameters or physical-mm calibration constants here.


def clean_measurement_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.ndim != 2:
        raise ValueError('2D binary foot mask required')
    binary = np.uint8(mask > 0) * 255
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# ========================================================================
# ===== L07 Outlier Rejection / 여러 사진의 측정값 통합 =====
# ========================================================================

# Observation-level multi-image 2D measurement fusion.
#
# Independently processes the physical measurements produced by a classical OpenCV
# OpenCV pipeline. Does NOT average raw photo pixels, force a target shoe size,
# or use any calibration from a subject's known foot length.


@dataclass(frozen=True)
class Observation:
    view_id: str
    kind: str         # 'length' or 'width'
    mm: float
    paper_valid: bool
    foot_valid: bool
    view_valid: bool
    geometric_error_px: float = 0.0


@dataclass
class FusionResult:
    accepted_mm: float | None
    status: str
    used_views: list[str]
    rejected: dict[str,str]
    spread_mm: float | None
    observations: dict[str,float]

    def as_dict(self):return asdict(self)


def fuse(observations: Sequence[Observation],*,min_consensus=2)->FusionResult:
    """Conservative fusion after independent A4, silhouette and pose gates.

    `min_consensus` is number of *valid independent photos*, never photo
    pixels. An inlier is compared against MAD and a fixed small numerical
    tolerance (3 mm) for nearly identical candidates, not any known foot size.
    The spread is an inter-observation disagreement, not a statistical CI.
    """
    if min_consensus<2:raise ValueError('at least two independently valid views required')
    rejects={};accepted=[]
    for o in observations:
        if not np.isfinite(o.mm) or o.mm<=0:
            rejects[o.view_id]='nonfinite_or_nonpositive_measurement'
        elif not o.paper_valid:
            rejects[o.view_id]='paper_not_validated'
        elif not o.foot_valid:
            rejects[o.view_id]='foot_not_validated'
        elif not o.view_valid:
            rejects[o.view_id]='view_not_validated'
        elif not np.isfinite(o.geometric_error_px) or o.geometric_error_px<0:
            rejects[o.view_id]='invalid_geometry_error'
        else:accepted.append(o)
    if len(accepted)<min_consensus:
        return FusionResult(None,'insufficient_valid_views',[],rejects,None,{o.view_id:o.mm for o in observations})
    kinds={o.kind for o in accepted}
    if len(kinds)!=1:
        raise ValueError('do not mix length/width; fuse separately')
    values=np.asarray([o.mm for o in accepted],float)
    centre=float(np.median(values))
    mad=float(np.median(np.abs(values-centre)))
    threshold=max(3.,3.*1.4826*mad)
    candidates=[o for o in accepted if abs(o.mm-centre)<=threshold]
    for o in accepted:
        if o not in candidates:rejects[o.view_id]='inconsistent_with_other_views'
    if len(candidates)<min_consensus:
        return FusionResult(None,'no_stable_consensus',[],rejects,None,{o.view_id:o.mm for o in observations})
    # Reject an ambiguous two-cluster set even if MAD would be too permissive.
    # A 2-view group cannot validate an outlier; differences > 12 mm are not fused.
    arr=np.array([o.mm for o in candidates])
    if len(candidates)==2 and np.ptp(arr)>12:
        return FusionResult(None,'two_views_disagree',[],rejects,float(np.ptp(arr)),{o.view_id:o.mm for o in observations})
    # Weight geometrically well-validated observations, but cap its influence
    # so a single photo cannot dominate a consensus.
    errors=np.array([o.geometric_error_px for o in candidates],float)
    weights=np.clip(1./(1.+errors/2.0),0.25,1.)
    weights/=weights.sum()
    estimate=float(np.sum(weights*arr))
    spread=float(np.ptp(arr)) if len(arr)>1 else 0.
    return FusionResult(estimate,'provisional' if len(candidates)<3 else 'cross_view_consistent',
        [o.view_id for o in candidates],rejects,spread,{o.view_id:o.mm for o in observations})


def fuse_with_reference_evidence(observations: Sequence[Observation],
                                 verified_ids: set[str],
                                 *, min_consensus=2) -> FusionResult:
    """L07 robust fusion that prefers independently located A4 borders.

    Distinguishes a sheet located by four outer edges from a mere bounding
    box of black printed squares.  No person's measured foot size is used.

    - At least two verified A4 views: robustly fuse verified views only.
    - Exactly one verified view: use it as an anchor; a small number of other
      photo measurements may corroborate it if their 2D projections agree.
    - No verified border: retain the original provisional multi-view fusion.

    The relative agreement gate (3.5%) is a heuristic image-repeatability
    check, not a promise of physical-mm accuracy or foot-height correction.
    """
    observations=list(observations)
    trusted=[o for o in observations if o.view_id in verified_ids and
             o.paper_valid and o.foot_valid and o.view_valid and np.isfinite(o.mm)
             and o.mm>0 and np.isfinite(o.geometric_error_px)]
    if not trusted:
        return fuse(observations,min_consensus=min_consensus)
    rejected={}
    by_id={o.view_id:o for o in observations}
    for o in observations:
        if o not in trusted and (not o.view_valid or not o.paper_valid or not o.foot_valid
                                 or not np.isfinite(o.mm) or o.mm<=0):
            rejected[o.view_id]='invalid_or_unmeasurable_view'
    if len(trusted)>=min_consensus:
        result=fuse(trusted,min_consensus=min_consensus)
        for o in observations:
            if o.view_id not in verified_ids and o.view_id not in rejected:
                rejected[o.view_id]='lower_reference_evidence'
        rejected.update(result.rejected)
        result.rejected=rejected
        result.observations={o.view_id:o.mm for o in observations}
        result.status='verified_border_multi_view_provisional' if result.accepted_mm is not None else result.status
        return result

    # Single verified border. A less reliable projected measurement can
    # support the anchor, but never outweigh it or force it to a known value.
    anchor=trusted[0]
    tolerance=max(6.0, .035*anchor.mm)
    supports=sorted((o for o in observations if o.view_id not in verified_ids
                     and o.view_id not in rejected and np.isfinite(o.mm)
                     and o.mm>0 and o.view_valid and o.paper_valid and o.foot_valid
                     and abs(o.mm-anchor.mm)<=tolerance),
                    key=lambda o:(abs(o.mm-anchor.mm),o.view_id))[:2]
    selected=[anchor]+supports
    for o in observations:
        if o.view_id not in rejected and o not in selected:
            rejected[o.view_id]='inconsistent_with_verified_paper_view'
    # Give the outer-border-verified source at least half the weight.  The
    # corroborating uncertain photos have equal capped weight in total.
    weights=np.array([2.0]+[1.0]*len(supports),dtype=float)
    measures=np.array([o.mm for o in selected],dtype=float)
    mm=float(np.average(measures,weights=weights))
    spread=float(np.ptp(measures)) if len(selected)>=2 else None
    return FusionResult(mm,
        'verified_border_with_corrob_provisional' if supports else 'single_verified_border_provisional',
        [o.view_id for o in selected],rejected,spread,
        {o.view_id:o.mm for o in observations})


# ========================================================================
# ===== L03 색 분리 + L05 Homography / 발볼·발길이 계산 =====
# ========================================================================

# Standalone conservative A4 reference + HSV/Lab foot segmentation (2D).
#
# This is a REBUILT estimator, not an anatomical 3D or clinically validated length estimator.
# Projected length is a floor-plane approximation, not actual certified anatomy.

@dataclass
class FrameAnalysis:
    view_id: str
    kind: str
    accepted: bool
    mm: float | None
    reason: str
    corner_score: float | None
    refined: bool
    paper_corners: list | None
    mask_pixels: int
    image: np.ndarray | None=None
    paper_method: str | None=None
    outer_border_verified: bool=False
    detected_squares: int=0
    rectified_image: np.ndarray | None=None
    mask_image: np.ndarray | None=None
    checker_quality: dict | None=None
    outer_edge_audit: dict | None=None
    orientation_corrected: bool=False
    checker_scale_excluded: bool=False
    partial_border_diagnostics: dict | None=None
    reference_transfer_evidence: dict | None=None

    def json(self):
        return {f.name:getattr(self,f.name) for f in fields(self) if f.name not in ('image','rectified_image','mask_image')}



def top_width(mask,px_per_mm=2.):
    """Broad forefoot cross-sections instead of calf or individual toes."""
    mask=clean_measurement_mask(mask)
    h,w=mask.shape
    rows=[]
    for y in range(int(h*.14),int(h*.63)):
        xx=np.where(mask[y]>0)[0]
        if len(xx)>=.20*w:
            # A row spanning paper borders likely includes floor, not foot.
            if xx.min()<3 or xx.max()>w-4:continue
            rows.append(float((xx[-1]-xx[0])/px_per_mm))
    if len(rows)<30:return None,'forefoot_not_separable'
    widths=np.array(rows)
    # Avoid single-pixel spikes and choose repeat-supported broad region.
    width=float(np.percentile(widths,95))
    if not (35<=width<=165):return None,'width_out_of_physical_range'
    return width,'projected_forefoot_width'


def side_length(mask,px_per_mm=2.):
    """Toe-heel silhouette extent on A4 plane; biased by foot height, not ground truth."""
    mask=clean_measurement_mask(mask)
    h,w=mask.shape
    yy,xx=np.where(mask>0)
    good=(yy>h*.30)&(yy<h*.91)
    xy=xx[good]
    if len(xy)<.045*w*h:return None,'insufficient_visible_foot'
    # Repeated-column support retains true toe/heel boundary while rejecting
    # isolated single-pixel noise. Quantile trimming cut real tips in photos.
    occupancy=np.bincount(xy,minlength=w)
    repeated=np.where(occupancy>=6)[0]
    if len(repeated)<50:return None,'insufficient_endpoint_support'
    left=float(repeated[0]);right=float(repeated[-1])
    length=(right-left)/px_per_mm
    if left<=3 or right>=w-4:
        return None,'toe_or_heel_at_paper_border'
    if not (120<=length<=335):return None,'length_out_of_physical_range'
    return length,'floor_plane_projected_extent_not_physical_length'


def analyze_frame(bgr,kind,view_id):
    if kind not in ('top','side'):raise ValueError('kind must be top or side')
    im=resize_image(bgr)
    try:
        quad,info=detect_paper(im)
        rect,hom,dim=rectify(im,quad)
        mask=foot_mask(rect)
        rect, mask, orientation_corrected = normalize_measurement_orientation(rect, mask, kind)
        if kind=='top':
            if dim[1]<=dim[0]:
                mm,reason=None,'paper_orientation_not_suitable_for_top'
            else:mm,reason=top_width(mask)
        else:
            if dim[0]<=dim[1]:
                mm,reason=None,'paper_orientation_not_suitable_for_side'
            else:mm,reason=side_length(mask)
        overlay=im.copy()
        cv2.polylines(overlay,[np.rint(quad).astype(np.int32)],True,(0,170,255),3)
        cv2.putText(overlay,f'A4 {info["method"]} / {"OK" if mm else "NO METRIC"}',(12,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        frame = FrameAnalysis(view_id,kind,mm is not None,round(mm,2) if mm is not None else None,reason,info['score'],info['refined'],quad.round(1).tolist(),int(np.count_nonzero(mask)),overlay,info.get('method'),bool(info.get('outer_border_verified',False)),int(info.get('detected_squares',0)),rect,mask)
        frame.orientation_corrected = orientation_corrected
        frame.partial_border_diagnostics = info.get('partial_border_diagnostics')
        frame.checker_quality = audit_checker_squares(rect).json()
        # Only record independently visible paper-to-floor edges. A black-square
        # bounding box is NOT reclassified as a fully verified A4 border.
        try:
            edges = audit_outer_edges(im, quad)
            frame.outer_edge_audit = audit_summary(edges)
            partial = info.get('partial_border_diagnostics')
            if partial and partial.get('applied'):
                earlier = np.asarray(info['checkerboard_provisional_corners'], np.float32)
                # Orange = provisional checkerbox; green = observed paper edge;
                # blue = bounded partial A4 refit. Missing sides are not verified.
                frame.image = draw_border_refinement(im, earlier, quad, partial)
            else:
                frame.image = draw_outer_edge_audit(overlay, quad, edges)
        except (ValueError, cv2.error):
            # Nonessential diagnostics must never turn an otherwise valid foot
            # measurement into an error or silently alter its millimeter value.
            frame.outer_edge_audit = None
        return frame
    except (DetectionFailure,cv2.error,ValueError) as exc:
        return FrameAnalysis(view_id,kind,False,None,str(exc),None,False,None,0,im)


# ========================================================================
# L02 Contours + L05 Homography + L07 RANSAC
# 측면 사진: 같은 사진 묶음에서 A4 바깥 경계가 독립 검출된 기준 사진을
# 이용해 다른 측면 사진의 검은 칸 중심점을 2D 대응. 인쇄 칸 크기 가정 없음.
# ========================================================================

def transfer_reference_corners(ref_image, ref_corners, image, old_corners):
    """Return (new A4 quad, evidence) only if >=5 feature points corroborate it.

    Feature correspondences are matched in the approximate rectified A4 frame,
    then re-fitted in raw image coordinates with RANSAC.  Require spatial
    coverage and independent paper/floor edge contrast on ALL FOUR edges.
    A transfer is not independent physical page-border verification and never
    uses the person's known foot size. No learned models or new markers.
    """
    source = resize_image(ref_image)
    target = resize_image(image)
    refq = order_quad(np.asarray(ref_corners, np.float32))
    oldq = order_quad(np.asarray(old_corners, np.float32))
    source_squares = detect_squares(source)
    target_squares = detect_squares(target)
    if len(target_squares) < 5:
        target_squares = detect_squares_adaptive(target)
    if len(source_squares) < 5 or len(target_squares) < 5:
        return None
    _, src_h, src_dim = rectify(source,refq)
    _, tgt_h, tgt_dim = rectify(target,oldq)
    if src_dim != (297.0,210.0) or tgt_dim != (297.0,210.0):
        return None
    src_pts = np.float32([s['center'] for s in source_squares]).reshape(-1,1,2)
    tgt_pts = np.float32([s['center'] for s in target_squares]).reshape(-1,1,2)
    a = cv2.perspectiveTransform(src_pts,src_h).reshape(-1,2)
    b = cv2.perspectiveTransform(tgt_pts,tgt_h).reshape(-1,2)
    distances = np.linalg.norm(b[:,None,:]-a[None,:,:],axis=2)
    matches=[];used_src=set();used_tgt=set()
    for i,j in sorted(np.ndindex(distances.shape),key=lambda ij:float(distances[ij])):
        if distances[i,j]>70.:break
        if i in used_tgt or j in used_src:continue
        used_src.add(j);used_tgt.add(i)
        matches.append((i,j))
    if len(matches)<5:
        return None
    src_idx=[j for i,j in matches]
    tgt_idx=[i for i,j in matches]
    reference_xy=a[src_idx]
    if np.ptp(reference_xy[:,0])<230 or np.ptp(reference_xy[:,1])<130:
        return None
    obj_xy=src_pts[src_idx,0]
    img_xy=tgt_pts[tgt_idx,0]
    hom,inlier=cv2.findHomography(obj_xy,img_xy,cv2.RANSAC,5.0,maxIters=2000)
    if hom is None or inlier is None or int(inlier.sum())<5:
        return None
    h=np.asarray(hom,dtype=np.float64)
    if abs(np.linalg.det(h))<1e-10:return None
    projected=cv2.perspectiveTransform(refq.reshape(-1,1,2),hom).reshape(4,2)
    try: candidate=validate_a4_quadrilateral(projected,target.shape)
    except DetectionFailure:return None
    # Prevent far-off and folded/perspective-degenerate proposals.
    H,W=target.shape[:2]
    if (candidate.min()<4 or candidate[:,0].max()>W-5 or candidate[:,1].max()>H-5):
        return None
    if cv2.contourArea(candidate)<=0:
        return None
    area=cv2.contourArea(candidate)/max(cv2.contourArea(oldq.astype(np.float32)),1.)
    shifts=np.linalg.norm(candidate-oldq,axis=1)
    if not .82<area<1.23 or np.max(shifts)>80:
        return None
    predicted=cv2.perspectiveTransform(obj_xy.reshape(-1,1,2),hom).reshape(-1,2)
    residual=np.linalg.norm(predicted-img_xy,axis=1)
    valid=np.asarray(inlier).reshape(-1)>0
    if float(np.median(residual[valid]))>3.5:
        return None
    # Distinct from repeated checker matching: the projected paper boundary
    # must be independently supported by original photo paper/floor contrast.
    support=_edge_contrast(target,candidate)
    if support is None or min(support)<.30 or np.median(support)<.45:
        return None
    return candidate,dict(status='cross_photo_supported_not_independent',
                          method='l02_checker_centers_l05_homography_l07_ransac',
                          inliers=int(inlier.sum()),matches=len(matches),
                          median_reprojection_px=round(float(np.median(residual[valid])),2),
                          observed_boundary_support=[round(float(v),2) for v in support],
                          original_candidate_shift_px=[round(float(v),1) for v in shifts])


def improve_side_with_verified_reference(items,sides):
    """Revise only reference-transfer candidates verified by separate cues."""
    verified=[f for f in items if f.kind=='side' and f.outer_border_verified
              and f.paper_corners is not None and f.accepted]
    if not verified or not sides:
        return items
    verified.sort(key=lambda f:(f.detected_squares,f.corner_score or 0),reverse=True)
    reference=verified[0]
    index=int(reference.view_id.split('-')[-1])-1
    source=sides[index]
    for item in items:
        if item.kind!='side' or item.outer_border_verified or item.paper_corners is None:
            continue
        i=int(item.view_id.split('-')[-1])-1
        match=transfer_reference_corners(source,reference.paper_corners,
                                         sides[i],item.paper_corners)
        if match is None: continue
        quad,evidence=match
        target=resize_image(sides[i])
        rect,_,dim=rectify(target,quad)
        if dim[0]<=dim[1]:continue
        mask=foot_mask(rect)
        rect,mask,rotated=normalize_measurement_orientation(rect,mask,'side')
        mm,reason=side_length(mask)
        if mm is None: continue
        # Require the transfer itself to be supported by checkerboard geometry
        # after rectification; do not certify it as an independently observed A4.
        checker=audit_checker_squares(rect).json()
        if checker.get('status') != 'supported':continue
        item.mm=round(mm,2)
        item.accepted=True
        item.reason=reason
        item.paper_corners=quad.round(1).tolist()
        item.paper_method='cross_photo_checker_ransac_unverified_outer_border'
        item.outer_border_verified=False
        item.reference_transfer_evidence={'reference':reference.view_id, **evidence}
        item.rectified_image=rect
        item.mask_image=mask
        item.mask_pixels=int(np.count_nonzero(mask))
        item.orientation_corrected=rotated
        item.checker_quality=checker
        original=target.copy()
        cv2.polylines(original,[np.rint(np.asarray(item.paper_corners,np.float32)).astype(np.int32)],True,(30,225,60),3)
        cv2.putText(original,'A4: checker RANSAC + photo edge checks (provisional)',
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2,cv2.LINE_AA)
        item.image=original
    return items


def analyze_batch(tops,sides):
    """All photos contribute to ONE physically scoped measurement per kind."""
    items=[]
    for i,im in enumerate(tops):items.append(analyze_frame(im,'top',f'top-{i+1}'))
    for i,im in enumerate(sides):items.append(analyze_frame(im,'side',f'side-{i+1}'))
    improve_side_with_verified_reference(items,sides)
    # Check repeated printed-square sizes across photos without using physical
    # foot measurements or a made-up checker print template. Diagnostic only.
    # CheckerQuality는 위의 체스보드 검사 구역에서 이미 정의됨.
    qualities=[CheckerQuality(**f.checker_quality) if f.checker_quality else
               CheckerQuality(0,0,None,None,'not_available') for f in items]
    for frame, quality in zip(items, compare_photos(qualities)):
        frame.checker_quality = quality
        # L07 outlier rejection: the same printed A4 should yield similarly
        # sized black squares *after* each independent Homography. The existing
        # checkerboard_quality compare_photos() only flags deviations when
        # >=3 photos supply suitable squares. Reject only a strong, supported
        # cross-photo conflict; this does NOT establish absolute calibration.
        if (frame.accepted and
                quality.get('status') == 'cross_photo_scale_disagreement'):
            frame.accepted = False
            frame.reason = 'checkerboard_scale_inconsistent'
            frame.checker_scale_excluded = True
    res={}
    for kind,key in [('top','width'),('side','length')]:
        obs=[Observation(o.view_id,key,o.mm if o.mm is not None else 0.,o.corner_score is not None,o.mask_pixels>0,o.accepted,0.) for o in items if o.kind==kind]
        # All values remain PROJECTION estimates; fusion never removes height parallax.
        verified_ids={f.view_id for f in items if f.kind==kind and f.accepted and f.outer_border_verified}
        r=fuse_with_reference_evidence(obs,verified_ids)
        result=r.as_dict()
        for excluded in items:
            if excluded.kind == kind and excluded.checker_scale_excluded:
                result['rejected'][excluded.view_id] = 'checkerboard_scale_inconsistent'
        used=[f for f in items if f.view_id in result['used_views']]
        result['reference_status'] = (
            'mixed_verified_and_unverified' if any(f.outer_border_verified for f in used) and any(not f.outer_border_verified for f in used)
            else 'unverified_checkerboard_bbox' if any(not f.outer_border_verified for f in used)
            else 'outer_border_heuristic' if used else 'unavailable'
        )
        result['verified_border_views'] = sorted(verified_ids)
        result['is_physical_accuracy_verified'] = False
        res[key]=result
    return items,res


# ========================================================================
# ===== L02 검은 칸 Contours + L07 ORB·RANSAC / 사진 간 검사 =====
# ========================================================================

# L07 ORB matching + RANSAC Homography, on *existing* checkerboard A4.
#
# Diagnostic only.  The repeating checkerboard has ambiguous visual features,
# so a fitted H does NOT provide independent A4 paper corners or millimetres.
# This module never changes the single-photo estimator or fused foot sizes.


@dataclass(frozen=True)
class PairEvidence:
    source: str
    target: str
    status: str
    matches: int
    inliers: int
    inlier_ratio: float | None
    source_coverage: list[float] | None
    target_coverage: list[float] | None
    candidate_corner_delta_px: list[float] | None
    note: str = 'Same-paper relative check only: not independently verified A4 corners.'
    method: str = 'orb_ransac'

    def json(self):
        return asdict(self)


def _features(image: np.ndarray, quad: np.ndarray, *, nfeatures: int = 2200):
    """L02 Contours + Morphology restrict L07 ORB to detected paper squares.

    The restriction avoids stable-but-wrong matches to room floor joints,
    clothes or skin.  No checkerboard cell pitch is assumed.
    """
    img = resize_image(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for square in detect_squares(img):
        corners = np.asarray(square.get('corners'), np.float32)
        if corners.shape == (4, 2):
            cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
    mask = cv2.dilate(mask, np.ones((23,23),np.uint8))
    page = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(page, np.rint(order_quad(quad)).astype(np.int32), 255)
    mask = cv2.bitwise_and(mask,page)
    # Suppress saturated skin/clothes inside the dilated checker ROI; a true
    # monochrome printed square and surrounding paper are low-saturation.
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_and(mask, np.uint8(hsv[:,:,1] < 72)*255)
    orb = cv2.ORB_create(nfeatures=nfeatures,scaleFactor=1.2,nlevels=8,
                         fastThreshold=7,edgeThreshold=10,patchSize=21)
    return orb.detectAndCompute(gray, mask)


def _ratio_matches(d1, d2, *, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = bf.knnMatch(d1, d2, k=2)
    backward = bf.knnMatch(d2, d1, k=2)
    reverse = {match.queryIdx: match.trainIdx for pair in backward if len(pair)==2
               for match, other in [pair] if match.distance < ratio*other.distance}
    return [match for pair in forward if len(pair)==2
            for match, other in [pair] if match.distance < ratio*other.distance
            and reverse.get(match.trainIdx) == match.queryIdx]


def _coverage(points: np.ndarray, quad):
    size = np.ptp(order_quad(quad),axis=0)
    return [round(float(x),3) for x in np.ptp(points,axis=0)/np.maximum(size,1.)]


def compare_same_board(source_img, target_img, src_quad, dst_quad,
                       *, source_id='source', target_id='target',
                       min_inliers=24):
    """L07 image stitching pattern: ORB -> ratio test -> H(RANSAC).

    Accept a correspondence audit only when spatial coverage AND inlier
    support are strong.  Bad RANSAC on a repeating pattern is inconclusive,
    never an instruction to move physical paper corners.
    """
    empty = lambda status, nm=0, ni=0, ratio=None, c0=None, c1=None: PairEvidence(
        source_id,target_id,status,nm,ni,ratio,c0,c1,None)
    if src_quad is None or dst_quad is None:
        return empty('insufficient_reference')
    q0 = order_quad(src_quad).astype(np.float32)
    q1 = order_quad(dst_quad).astype(np.float32)
    k0,d0 = _features(source_img,q0)
    k1,d1 = _features(target_img,q1)
    matches = _ratio_matches(d0,d1)
    if len(matches) < min_inliers:
        return empty('insufficient_correspondence',len(matches))
    src = np.float32([k0[m.queryIdx].pt for m in matches])
    dst = np.float32([k1[m.trainIdx].pt for m in matches])
    try:
        H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0,
                                           maxIters=2500,confidence=.995)
    except cv2.error:
        return empty('ransac_failed',len(matches))
    if H is None or inlier_mask is None or not np.isfinite(H).all():
        return empty('ransac_failed',len(matches))
    ok = inlier_mask.ravel().astype(bool)
    count = int(np.count_nonzero(ok))
    frac = round(count/len(matches),3)
    if count < 4:
        return empty('insufficient_inliers',len(matches),count,frac)
    coverage0 = _coverage(src[ok],q0)
    coverage1 = _coverage(dst[ok],q1)
    if (count < min_inliers or frac < .45 or min(*coverage0,*coverage1) < .34):
        return empty('insufficient_spatial_support',len(matches),count,frac,coverage0,coverage1)
    predicted = cv2.perspectiveTransform(q0.reshape(1,4,2),H).reshape(4,2)
    if not np.isfinite(predicted).all() or not cv2.isContourConvex(predicted.astype(np.float32)):
        return empty('degenerate_homography',len(matches),count,frac,coverage0,coverage1)
    # This is only a relative discrepancy between two automatic A4 guesses.
    # It must NEVER be silently presented as mm error or exact true corners.
    drift = [round(float(x),1) for x in np.linalg.norm(predicted-q1,axis=1)]
    status = 'relative_geometry_consistent' if max(drift)<=12.0 else 'relative_geometry_disagreement'
    return PairEvidence(source_id,target_id,status,len(matches),count,frac,
                        coverage0,coverage1,drift)



def _square_centers_on_paper(image, quad):
    """L02 Contours -> L05 planar warp of only observed checker centers.

    We use the image-space A4 candidate to normalize photo orientation for
    matching. This is a RELATIVE layout check and cannot establish print pitch,
    margins, true paper corners or any millimetre calibration.
    """
    img = resize_image(image)
    q = order_quad(quad).astype(np.float32)
    sides=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
    width,height=(594,420) if sides[0]+sides[2]>=sides[1]+sides[3] else (420,594)
    dst=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    H=cv2.getPerspectiveTransform(q,dst)
    features=[s for s in detect_squares(img)
              if cv2.pointPolygonTest(q,tuple(map(float,s['center'])),False)>=0]
    if not features:
        return np.empty((0,2),np.float32),np.empty((0,2),np.float32),(width,height)
    points=np.asarray([s['center'] for s in features],np.float32).reshape(-1,1,2)
    normalized=cv2.perspectiveTransform(points,H).reshape(-1,2)
    return points.reshape(-1,2),normalized,(width,height)


def compare_checker_centers(source_img,target_img,src_quad,dst_quad,
                            *,source_id='source',target_id='target',
                            min_inliers=5):
    """Correspond original L02 checker centers using L05 / L07 RANSAC.

    Unlike ORB, this uses only automatically observed black-square centers,
    excluding foot and floor descriptors. It is deliberately not a calibration
    or a method of filling missing paper corners.
    """
    def report(status,nm=0,ni=0,ratio=None,c0=None,c1=None,delta=None):
        return PairEvidence(source_id,target_id,status,nm,ni,ratio,c0,c1,delta,
            'Matched black-square centers on the same provisional A4 plane, not verified A4 corners.',
            'checker_centers_ransac')
    if src_quad is None or dst_quad is None:
        return report('insufficient_reference')
    q0=order_quad(src_quad).astype(np.float32)
    q1=order_quad(dst_quad).astype(np.float32)
    a,na,dim0=_square_centers_on_paper(source_img,q0)
    b,nb,dim1=_square_centers_on_paper(target_img,q1)
    if dim0!=dim1:
        return report('incompatible_photo_orientation')
    if len(a)<min_inliers or len(b)<min_inliers:
        return report('insufficient_correspondence')
    distance=np.linalg.norm(na[:,None,:]-nb[None,:,:],axis=2)
    match_a=distance.argmin(axis=1)
    match_b=distance.argmin(axis=0)
    # L07: one-to-one image correspondence and outlier rejection. The radius
    # is in normalized A4 IMAGE pixels, not a presumed checker-square pitch.
    matched=[(i,int(j)) for i,j in enumerate(match_a)
             if match_b[j]==i and distance[i,j] <= 25.0]
    if len(matched)<min_inliers:
        return report('insufficient_correspondence',len(matched))
    src=np.float32([a[i] for i,j in matched]); dst=np.float32([b[j] for i,j in matched])
    try:
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.0,maxIters=2500,confidence=.995)
    except cv2.error:
        return report('ransac_failed',len(matched))
    if H is None or mask is None or not np.isfinite(H).all():
        return report('ransac_failed',len(matched))
    inside=mask.ravel().astype(bool)
    count=int(inside.sum())
    ratio=round(count/len(matched),3)
    if count<min_inliers:
        return report('insufficient_inliers',len(matched),count,ratio)
    c0=_coverage(src[inside],q0)
    c1=_coverage(dst[inside],q1)
    if ratio<.70 or min(*c0,*c1)<.48:
        return report('insufficient_spatial_support',len(matched),count,ratio,c0,c1)
    projected=cv2.perspectiveTransform(q0.reshape(1,4,2),H).reshape(4,2)
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected.astype(np.float32)):
        return report('degenerate_homography',len(matched),count,ratio,c0,c1)
    delta=[round(float(x),1) for x in np.linalg.norm(projected-q1,axis=1)]
    if max(delta)>12:
        status='relative_geometry_disagreement'
    elif count<7:
        status='relative_geometry_consistent_limited'
    else:
        status='relative_geometry_consistent'
    return report(status,len(matched),count,ratio,c0,c1,delta)


def audit_same_view_photos(tops: Sequence[np.ndarray], sides: Sequence[np.ndarray],
                           frames: Sequence) -> dict:
    """A compact diagnostic per kind: max 7 comparisons, O(n) matches.

    Prefer an outer-border-verified photo as an anchor; otherwise use the
    photo with most visible checker squares. This is not ground truth.
    """
    items = list(frames)
    results = {}
    for kind, images in (('top',tops),('side',sides)):
        group = [(i,f) for i,f in enumerate(items) if f.kind==kind]
        available = [(i,f) for i,f in group if f.paper_corners is not None]
        if len(available)<2:
            results[kind]={'anchor':None,'pairs':[], 'note':'Not enough detectable A4s'}
            continue
        anchor_i,anchor=max(available, key=lambda it: (
            bool(it[1].outer_border_verified), it[1].detected_squares, -it[0]))
        source = images[int(anchor.view_id.split('-')[-1])-1]
        pairs=[]
        for _,f in available:
            if f.view_id==anchor.view_id:continue
            target = images[int(f.view_id.split('-')[-1])-1]
            # Use checker centers as the primary correspondence, because an
            # ORB patch around a printed square may still contain foot skin.
            primary=compare_checker_centers(source,target,anchor.paper_corners,f.paper_corners,
                    source_id=anchor.view_id,target_id=f.view_id)
            orb=compare_same_board(source,target,anchor.paper_corners,f.paper_corners,
                    source_id=anchor.view_id,target_id=f.view_id)
            entry=primary.json()
            entry['orb_crosscheck']={'status':orb.status,'inliers':orb.inliers,
                                     'matches':orb.matches}
            pairs.append(entry)
        results[kind]={'anchor':anchor.view_id,'pairs':pairs,
            'note':'L02 Contours + L05 Homography + L07 RANSAC; optional ORB corroboration. No A4 physical calibration.'}
    return results


# ========================================================================
# ===== L02 Contours·마스크 / 실제 측정선 표시 =====
# ========================================================================

# L02 contours/masks and L05 A4 plane: transparent 2D measurement overlay.
#
# Draw exactly the pixel locations used by core.measurement.top_width/side_length.
# These are projected silhouette extrema, not independently detected anatomical
# heel or toe landmarks. No manual click, foot-size constant, or 3D correction.


def _forefoot_rows(mask):
    mask=clean_measurement_mask(mask)
    h, w = mask.shape
    candidates = []
    for y in range(int(h * .14), int(h * .63)):
        xx = np.where(mask[y] > 0)[0]
        if len(xx) >= .20 * w and xx.min() >= 3 and xx.max() <= w - 4:
            candidates.append((int(y), int(xx[0]), int(xx[-1])))
    return candidates


def _side_endpoint_locations(mask):
    mask=clean_measurement_mask(mask)
    h, w = mask.shape
    yy, xx = np.where(mask > 0)
    active = xx[(yy > h * .30) & (yy < h * .91)]
    if len(active) < .045 * w * h:
        return None
    support = np.bincount(active, minlength=w)
    valid = np.where(support >= 6)[0]
    if len(valid) < 50:
        return None
    x0, x1 = int(valid[0]), int(valid[-1])
    # The measurement is the horizontal difference x1-x0. Mark the *actual*
    # silhouette pixels that supplied the extremes rather than drawing fake
    # heel/toe landmarks along an empty row of the image.
    ys0 = yy[(xx >= x0) & (xx <= x0 + 2) & (yy > h*.30) & (yy < h*.91)]
    ys1 = yy[(xx >= x1 - 2) & (xx <= x1) & (yy > h*.30) & (yy < h*.91)]
    if len(ys0)==0 or len(ys1)==0:
        return None
    return (x0, int(np.median(ys0)), x1, int(np.median(ys1)))


def _side_columns(mask):
    points = _side_endpoint_locations(mask)
    return (points[0], points[2]) if points is not None else None


def annotated_measurement(rectified, mask, kind, *, accepted, used_in_fusion,
                          millimeters=None, reason=None):
    """Render visual evidence based solely on the production pixel masks."""
    if rectified is None or mask is None:
        return None
    out = rectified.copy()
    h, w = mask.shape
    color = (50, 175, 45) if used_in_fusion else (35, 125, 235)
    status = 'IN FINAL RESULT' if used_in_fusion else 'NOT IN FINAL RESULT'
    if kind == 'top':
        rows = _forefoot_rows(mask)
        if len(rows) >= 30:
            widths = np.array([x1 - x0 for _, x0, x1 in rows], dtype=float)
            target = float(np.percentile(widths, 95))
            y, x0, x1 = min(rows, key=lambda r: abs(r[2] - r[1] - target))
            for yy, a, b in rows[::max(1, len(rows)//18)]:
                cv2.line(out, (a, yy), (b, yy), (180, 180, 70), 1, cv2.LINE_AA)
            cv2.line(out, (x0, y), (x1, y), color, 3, cv2.LINE_AA)
            cv2.circle(out, (x0, y), 5, color, -1)
            cv2.circle(out, (x1, y), 5, color, -1)
    elif kind == 'side':
        endpoints = _side_endpoint_locations(mask)
        if endpoints is not None:
            x0, y0, x1, y1 = endpoints
            # The dimension is horizontal (x1-x0), not the Euclidean distance
            # between two image points. The colored circles identify real
            # mask-supported silhouette extrema; the horizontal line is a
            # dimension guide, not an invented anatomical toe/heel landmark.
            line_y = min(h-26, max(y0,y1)+28)
            cv2.line(out, (x0,y0), (x0,line_y), (130,170,140), 1, cv2.LINE_AA)
            cv2.line(out, (x1,y1), (x1,line_y), (130,170,140), 1, cv2.LINE_AA)
            cv2.line(out, (x0,line_y), (x1,line_y), color, 3, cv2.LINE_AA)
            cv2.circle(out, (x0,y0), 5, color, -1)
            cv2.circle(out, (x1,y1), 5, color, -1)
    else:
        raise ValueError('top or side required')
    display_value = f'{millimeters:.1f} mm' if millimeters is not None else 'NO MEASUREMENT'
    cv2.rectangle(out, (0, 0), (w, 52), (30, 50, 44), -1)
    cv2.putText(out, f'{display_value}  |  {status}', (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, .53, (255,255,255), 1, cv2.LINE_AA)
    if reason and (not accepted or not used_in_fusion):
        cv2.putText(out, str(reason)[:49], (8, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1, cv2.LINE_AA)
    return out


# ========================================================================
# ===== L09 YOLO / 기존 발 아치 high·low·normal 독립 분류 =====
# ========================================================================

# Original user-supplied high/low/normal YOLO model (unchanged weights).
#
# Independent optional branch: L09 image classification. It does NOT supply
# foot width/length pixels or train/recalibrate the 2D measurement pipeline.
# Model loading is lazy so the OpenCV-only foot measurement still operates when
# PyTorch/Ultralytics are unavailable on lightweight hosts.


MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'arch'
MODEL_FILENAME = 'best.pt'
SHA256_FILE = MODEL_DIR / 'best.pt.sha256'
CONF = 0.25   # The original user's arch_photo_viewer.py setting.
LABELS = {'high', 'low', 'normal'}
_MODEL_LOCK = threading.RLock()


def model_file() -> Path:
    """Reconstruct the supplied .pt bytes from GitHub-friendly splits once.

    Reassembly is byte-exact and SHA256-verified; no fine tuning, conversion or
    modification of checkpoint content takes place.
    """
    path = MODEL_DIR / MODEL_FILENAME
    expected = SHA256_FILE.read_text(encoding='ascii').strip().split()[0].lower()
    if len(expected) != 64:
        raise RuntimeError('YOLO 모델 SHA256 검사 파일이 잘못되었습니다.')
    with _MODEL_LOCK:
        if path.is_file():
            if _sha256(path) == expected:
                return path
            # Never use corrupted or locally replaced model weights silently.
            raise RuntimeError('YOLO best.pt SHA256이 원본 모델과 다릅니다.')
        # Web uploads are most reliable with small model chunks. Reassemble
        # the original weights byte-for-byte in numerical part order.
        segments = sorted(MODEL_DIR.glob('best.pt.part[0-9][0-9]'))
        valid_names = [f'best.pt.part{i:02d}' for i in range(1, len(segments) + 1)]
        if not segments or [p.name for p in segments] != valid_names:
            raise RuntimeError('YOLO 모델 분할 파일이 누락되었거나 번호가 연속되지 않습니다.')
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.assembling')
        sha = hashlib.sha256()
        try:
            with temp.open('wb') as dest:
                for segment in segments:
                    with segment.open('rb') as src:
                        while True:
                            part = src.read(1024 * 1024)
                            if not part:
                                break
                            sha.update(part)
                            dest.write(part)
            if sha.hexdigest() != expected:
                raise RuntimeError('YOLO 모델 조각의 SHA256 검증에 실패했습니다.')
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path


def _memory_limit_mb() -> int | None:
    """Read the container cgroup limit without importing PyTorch.

    YOLOv8m-seg needs substantially more than the 512 MB available to
    free Render services. A RAM guard prevents a YOLO request from killing
    the otherwise working 2D CV web process. It does not claim that an
    instance above this limit will automatically have enough RAM.
    """
    for filename in ('/sys/fs/cgroup/memory.max',
                     '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            raw=Path(filename).read_text(encoding='ascii').strip()
            size=int(raw)
            if 0 < size < (1 << 55):
                return size // (1024*1024)
        except (OSError, ValueError):
            continue
    return None


def arch_capability() -> dict:
    """Configuration checks only; does NOT claim to have run YOLO inference."""
    chunks=sorted(MODEL_DIR.glob('best.pt.part[0-9][0-9]'))
    required=[f'best.pt.part{i:02d}' for i in range(1,8)]
    segments_ok=[f.name for f in chunks] == required and SHA256_FILE.is_file()
    installed=importlib.util.find_spec('ultralytics') is not None
    ram=_memory_limit_mb()
    min_ram=1536  # conservative runtime gate, NOT a measured RAM requirement
    if not segments_ok:
        code='model_parts_missing'
        explanation='YOLO 모델 조각 7개 또는 SHA256 파일을 찾지 못했습니다.'
    elif not installed:
        code='dependency_missing'
        explanation='ultralytics가 설치되지 않았습니다. YOLO용 Render 빌드 설정을 확인하세요.'
    elif ram is not None and ram < min_ram and os.getenv('FOOTFRAME_YOLO_ALLOW_LOW_MEMORY')!='1':
        code='memory_insufficient'
        explanation=f'서버 메모리 한도 {ram}MB에서는 PyTorch YOLO로 인해 서버가 종료될 수 있습니다. 별도 2GB 이상 인스턴스가 필요합니다.'
    else:
        code='ready_to_attempt'
        explanation='파일 및 라이브러리 조건 확인. 실제 모델 추론 성공 여부는 예측 실행 결과로 확인해야 합니다.'
    return dict(status=code, message=explanation, installed=installed,
                model_parts_present=segments_ok, memory_limit_mb=ram,
                actual_inference_verified=False, model_task='segment')


def predict_arch_checked(image: np.ndarray) -> dict:
    gate=arch_capability()
    if gate['status']!='ready_to_attempt':
        raise RuntimeError(gate['message'])
    return predict_arch(image)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load_model():
    try:
        from ultralytics import YOLO
    except (ImportError, OSError) as exc:
        raise RuntimeError('YOLO 실행 환경이 없습니다. requirements-yolo.txt를 설치해 주세요.') from exc
    model = YOLO(str(model_file()))
    if model.task not in ('detect', 'segment', 'classify'):
        raise RuntimeError('원본 YOLO 모델의 task가 분류/검출/분할 형식이 아닙니다.')
    names = model.names
    values = names.values() if isinstance(names, dict) else names
    if {str(v).strip().lower() for v in values} != LABELS:
        raise RuntimeError(f'원본 모델의 클래스 구성이 예상과 다릅니다: {names}')
    return model


def predict_arch(image_bgr: np.ndarray) -> dict:
    """Preserve original YOLO semantics (CPU, conf=.25, highest box/class).

    Model confidence and classification score are *not* physical measurement
    accuracy. An unsupported image must not be forced to a shoe size.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('발 아치 분석에는 BGR 컬러 사진이 필요합니다.')
    if min(image_bgr.shape[:2]) < 64:
        raise ValueError('발 아치 사진 해상도가 너무 낮습니다.')
    with _MODEL_LOCK:
        model = _load_model()
        # Ultralytics treats numpy.ndarray source images as OpenCV BGR.
        # Converting to RGB here would swap channels twice and corrupt colors.
        # PIL input to the original tkinter viewer was RGB; Ultralytics
        # internally converts PIL to BGR before applying its RGB preprocessing.
        prediction = model.predict(image_bgr, conf=CONF, device='cpu', verbose=False)[0]
    out = {'status': 'no_detections', 'class_name': None,
           'confidence': None, 'model_task': model.task,
           'score_is_accuracy': False, 'model_unchanged': True,
           'preview_bgr': image_bgr, 'mask_count': 0}
    if prediction.probs is not None:
        scores = prediction.probs.data.cpu().tolist()
        best = int(np.argmax(scores))
        score = float(scores[best])
        name = str(prediction.names[best]).lower()
        out.update(status='classified' if score >= CONF else 'low_confidence',
                   class_name=name, confidence=round(score, 4))
    elif prediction.boxes is not None and len(prediction.boxes) > 0:
        best = int(prediction.boxes.conf.argmax().item())
        score = float(prediction.boxes.conf[best].item())
        label = str(prediction.names[int(prediction.boxes.cls[best].item())]).lower()
        # Original checkpoint is a YOLOv8-seg model (Low / High / Normal).
        # Draw masks and boxes; do not silently misrepresent it as a
        # classification-only checkpoint.
        masks=getattr(prediction, 'masks', None)
        count=len(masks.data) if masks is not None else 0
        out.update(status='detected' if score >= CONF else 'low_confidence',
                   class_name=label, confidence=round(score, 4),
                   mask_count=count, preview_bgr=prediction.plot())
    return out


# ========================================================================
# ===== 웹 서버·HTML·사진 업로드·분석 API =====
# ========================================================================

# FootFrame rebuilt standalone 2D web app. Python stdlib HTTP + OpenCV.
#
# No Flask dependency: available offline and deployable with `python app.py`.
# Uploaded photographs are kept only in memory, not saved to disk.

MAX_BYTES=34_000_000
MAX_IMAGE_BYTES=12_000_000
MAX_PER_KIND=8
MAX_ARCH_IMAGES=1
# Guard memory spikes on free hosting. These are request resource limits, NOT
# physical foot-size constraints or calibration constants.
MAX_DECODED_IMAGE_PIXELS=20_000_000
MAX_DECODED_SIDE=6500
_ANALYSIS_SLOT=threading.BoundedSemaphore(1)
HOME=Path(__file__).parent

HTML=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FootFrame · 2D 발 치수 분석</title><style>
:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#182338;background:#f6f8fb}
*{box-sizing:border-box}body{margin:0}.nav{background:white;border-bottom:1px solid #e4e8f0;display:flex;align-items:center;justify-content:space-between;padding:23px max(5%,calc((100% - 980px)/2))}
.brand{font-size:24px;font-weight:850;letter-spacing:-1px}.brand em{color:#245dd2;font-style:normal}.tag{font-size:12px;color:#6d7b8d}
main{width:min(900px,92%);margin:50px auto 85px}h1{font-size:clamp(28px,6vw,42px);letter-spacing:-1.6px;margin:8px 0 18px}
.lead{color:#596779;line-height:1.8;margin-bottom:36px;font-size:16px}.eyebrow{font-size:12px;letter-spacing:1.6px;font-weight:800;color:#245dd2}
.card{background:#fff;border:1px solid #e3e8ef;box-shadow:0 10px 40px #172c5610;border-radius:22px;padding:30px;margin:17px 0}
h2{font-size:20px;margin:0 0 12px}p{line-height:1.7}.hint{font-size:13px;color:#66758b}.file{width:100%;padding:19px;background:#f6f9ff;border:1px dashed #90a9d2;border-radius:12px;font-size:15px}
.steps{display:grid;grid-template-columns:1fr 1fr;gap:18px}.note{background:#f1f6ff;border-radius:14px;padding:15px;color:#345071;font-size:13px;line-height:1.7}
button{border:0;background:#235fdf;color:white;font-size:17px;font-weight:800;padding:17px 30px;border-radius:13px;cursor:pointer;width:100%;margin-top:18px}button:disabled{opacity:.55;cursor:wait}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px}.metric{padding:23px;border:1px solid #e1e9f4;border-radius:15px;background:#f6f9ff}
.metric small{display:block;color:#67768d}.metric strong{display:block;font-size:31px;margin-top:8px;letter-spacing:-1px}.error{padding:16px;border-radius:12px;background:#fff3ec;color:#9b4a2c;font-size:14px;line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.frame{border:1px solid #e6e9ef;border-radius:13px;padding:12px}.frame img{display:block;width:100%;border-radius:10px;object-fit:contain}.frame h3{font-size:14px}.stage{margin:12px 0;padding:6px;background:#f8fbff;border:1px solid #e5edf5;border-radius:10px}.stage figcaption{text-align:center;font-size:12px;color:#587084;padding:6px 0 2px}.pill{color:#2461af;font-size:12px;font-weight:700}.bad{color:#ac6a39}.muted{color:#64748b;font-size:13px;line-height:1.6}
footer{color:#8290a4;font-size:12px;text-align:center;padding-bottom:45px}
@media(max-width:650px){.steps,.metrics{grid-template-columns:1fr}.card{padding:22px}main{margin-top:32px}}
</style></head><body><header class="nav"><span class="brand">Foot<em>Frame</em></span><span class="tag">2D Computer Vision</span></header>
<main><div class="eyebrow">AUTOMATIC 2D MEASUREMENT</div><h1>체스보드 A4로 시작하는<br>발 치수 분석</h1><p class="lead">발과 체스보드가 인쇄된 A4를 함께 촬영한 사진을 여러 장 업로드하면, 영상처리·색공간·Homography로 2D 투영치수를 계산합니다. 수동으로 꼭짓점을 찍을 필요가 없습니다.</p>
<div class="note">촬영: 체스보드 A4는 바닥에 그대로 두고, <b>위쪽 사진 2~8장</b>은 거의 수직으로, <b>측면 사진 2~8장</b>은 발 옆에서 찍어 주세요. 사진은 서버 메모리에서 분석하며 저장하지 않습니다. <b>사진 속 기존 체스보드 A4를 그대로 사용하며 다른 인쇄판은 필요하지 않습니다.</b></div>
<form id="form"><div class="steps"><div class="card"><h2>01 · 위쪽 사진</h2><p class="hint">발볼 폭 추정 · A4 네 모서리가 최대한 보이도록</p><input class="file" name="top" id="top" type="file" accept="image/jpeg,image/png,image/webp" multiple required><p class="hint" id="topCount">최대 8장</p></div>
<div class="card"><h2>02 · 측면 사진</h2><p class="hint">발 길이 투영 추정 · A4는 바닥에 그대로</p><input class="file" name="side" id="side" type="file" accept="image/jpeg,image/png,image/webp" multiple required><p class="hint" id="sideCount">최대 8장</p></div></div><div class="card"><h2>03 · YOLO 발 아치 판별 (선택)</h2><p class="hint">기존 high / low / normal YOLO 모델을 변경하지 않고 그대로 사용합니다. 발의 아치가 보이는 별도의 사진 1장을 추가할 수 있습니다. 아치 분류는 발 길이·발볼 측정에 영향을 주지 않습니다.</p><input class="file" name="arch" id="arch" type="file" accept="image/jpeg,image/png,image/webp"><p class="hint" id="archStatus">YOLO 실행 서버 환경 확인 중...</p><button type="button" id="archOnly" style="background:#285a57">발 아치 사진만 YOLO로 분석</button></div><button id="go" type="submit">자동 분석 시작</button></form>
<section id="result" style="display:none"><div class="card"><h2>통합 분석 결과</h2><div class="metrics"><div class="metric"><small>발볼 · A4 평면 투영</small><strong id="width">—</strong></div><div class="metric"><small>발 길이 · A4 평면 투영</small><strong id="length">—</strong></div></div><p class="muted" id="summary"></p><div class="error">중요: 길이는 바닥 평면 투영치입니다. 발의 실제 높이·카메라 렌즈·시점으로 인한 시차가 제거되지 않았으므로 <b>실제 신발 사이즈를 확정하는 용도로 사용하면 안 됩니다.</b></div></div>
<div class="card"><h2>사진별 자동 검출 진단</h2><div class="grid" id="frames"></div></div></section><div class="card" id="archSection" style="display:none"><h2>YOLO 발 아치 분류 (별도 분석)</h2><p class="muted" id="archMessage"></p><img id="archPreview" alt="YOLO 발 아치 결과" style="max-width:100%;max-height:360px;object-fit:contain;display:none;border-radius:12px"></div>
</main><footer>FootFrame Lecture-CV · OpenCV 2D · Checkerboard A4 · Provisional</footer>
<script>
const form=document.getElementById('form');const btn=document.getElementById('go');
for(const kind of ['top','side']) document.getElementById(kind).addEventListener('change',e=>document.getElementById(kind+'Count').textContent=e.target.files.length+'장 선택');
// Standalone YOLO input is possible without top/side photos.
function renderArch(a){
 const panel=document.getElementById('archSection');
 panel.style.display=a&&a.status!=='not_requested'?'block':'none';
 if(!a||a.status==='not_requested')return;
 const label=a.class_name?({high:'높은 아치',low:'낮은 아치',normal:'보통 아치'}[a.class_name]||a.class_name):'판단 보류';
 const score=a.confidence==null?'':` · YOLO 점수 ${(a.confidence*100).toFixed(1)}% (실제 측정 정확도 아님)`;
 document.getElementById('archMessage').textContent=a.status==='unavailable'?`YOLO 미실행: ${a.error||'환경을 확인해 주세요.'}`:`${label}${score} · 기존 가중치 사용`;
 const img=document.getElementById('archPreview');
 if(a.preview){img.src='data:image/jpeg;base64,'+a.preview;img.style.display='block';}
 else {img.style.display='none';img.removeAttribute('src');}
}
fetch('/arch-status').then(r=>r.json()).then(d=>{
 document.getElementById('archStatus').textContent=d.status==='ready_to_attempt'
   ?'YOLO 실행 준비됨 · 모델 추론 여부는 실제 사진으로 확인하세요.'
   :'YOLO 사용 불가: '+d.message;
}).catch(()=>{document.getElementById('archStatus').textContent='서버의 YOLO 실행 상태를 확인하지 못했습니다.'});
document.getElementById('archOnly').addEventListener('click',async()=>{
 const button=document.getElementById('archOnly');
 const file=document.getElementById('arch').files[0];
 if(!file){alert('발 아치 사진 한 장을 선택해 주세요.');return;}
 const body=new FormData();body.append('arch',file);
 button.disabled=true;button.textContent='YOLO 아치 분석 중...';
 try{const response=await fetch('/analyze-arch',{method:'POST',body});const data=await response.json();
     if(data.arch)renderArch(data.arch);else throw Error(data.error||'YOLO 실행 오류');}
 catch(error){renderArch({status:'unavailable',error:error.message});}
 finally{button.disabled=false;button.textContent='발 아치 사진만 YOLO로 분석';}
});
form.addEventListener('submit',async e=>{e.preventDefault();const tops=document.getElementById('top').files;const sides=document.getElementById('side').files;
if(!tops.length||!sides.length||tops.length>8||sides.length>8){alert('위쪽과 측면 사진을 각각 1~8장 선택해 주세요.');return;}
btn.disabled=true;btn.textContent='사진 분석 중...';
try{const r=await fetch('/analyze',{method:'POST',body:new FormData(form)});const data=await r.json();if(!r.ok)throw Error(data.error||'분석 오류');
const fmt=x=>x==null?'검출 불확실':x.toFixed(1)+' mm';
document.getElementById('width').textContent=fmt(data.fusion.width.accepted_mm);
document.getElementById('length').textContent=fmt(data.fusion.length.accepted_mm);
document.getElementById('summary').textContent=`발볼 사용 사진 ${data.fusion.width.used_views.length}장 · 측면 사진 A4 후보 ${data.side_detection_summary?.paper_candidate_found??0}/${data.side_detection_summary?.total??0}장 · 개별 길이 ${data.side_detection_summary?.individual_length_measured??0}장 · 최종 길이 사용 ${data.fusion.length.used_views.length}장. A4 후보 검출이 실제 종이 외곽과 일치한다는 뜻은 아닙니다.`;
const referenceWarning=[data.fusion.width,data.fusion.length].some(x=>['unverified_checkerboard_bbox','mixed_verified_and_unverified'].includes(x.reference_status));
if(referenceWarning)document.getElementById('summary').textContent+=' 일부 사진은 검은 칸을 이용해 A4 외곽을 추정한 것으로 실제 종이 네 모서리의 독립 검증이 없습니다.';
renderArch(data.arch);
const reasonKo={toe_or_heel_at_paper_border:'발가락 또는 뒤꿈치가 A4 경계에 걸려 측정 보류',insufficient_visible_foot:'발 윤곽이 충분히 보이지 않습니다',insufficient_endpoint_support:'발끝 검출 근거 부족',inconsistent_with_verified_paper_view:'외곽 검증 사진과 측정값 불일치',invalid_or_unmeasurable_view:'측정 불가능한 사진',checkerboard_scale_inconsistent:'다른 사진과 체스보드 칸 크기가 일치하지 않아 제외',checkerboard_not_found:'체스보드 검출 실패'};const rko=r=>reasonKo[r]||r||'확인 불가';
const g=document.getElementById('frames');g.replaceChildren();for(const f of data.frames){const el=document.createElement('div');el.className='frame';const h=document.createElement('h3');h.textContent=(f.source_filename||f.view_id)+' · '+(f.mm==null?'치수 계산 보류':f.mm.toFixed(1)+' mm');const metric=f.kind==='top'?data.fusion.width:data.fusion.length;const isUsed=metric.used_views.includes(f.view_id);const excludedReason=metric.rejected[f.view_id];const t=document.createElement('p');t.className=isUsed?'pill':'pill bad';t.textContent=isUsed?'최종 측정에 사용 · '+rko(f.reason):'최종 측정 제외 · '+rko(f.accepted?excludedReason:f.reason);el.append(h);for(const [label,value] of [['01 · A4 검출',f.overlay],['02 · Homography',f.rectified],['03 · Color Segmentation',f.mask_preview],['04 · 실제 측정선',f.measurement_preview]]){if(!value)continue;const figure=document.createElement('figure');figure.className='stage';const im=new Image();im.src='data:image/jpeg;base64,'+value;im.alt=f.view_id+' '+label;im.loading='lazy';const caption=document.createElement('figcaption');caption.textContent=label;figure.append(im,caption);el.append(figure)}
const diagnostic=document.createElement('p');diagnostic.className='muted';const quality=f.checker_quality||{};
const squareInfo=quality.status==='supported'?' · 원근 보정 후 검은 칸 '+quality.supported_squares+'개 반복 검출'
    :quality.status==='cross_photo_scale_disagreement'?' · 사진 간 검은 칸 크기 불일치 의심'
    :' · 검은 칸 크기 정합성 판단 보류';
const edgeAudit=f.outer_edge_audit||{};
const edgeNames={top:'위쪽',right:'오른쪽',bottom:'아래쪽',left:'왼쪽'};
const missing=Array.isArray(edgeAudit.unsupported_sides)?edgeAudit.unsupported_sides.map(x=>edgeNames[x]||x).join('·'):'';
const partial=f.partial_border_diagnostics;const fitted=partial?.applied?(' · Canny/RANSAC 외곽 부분 보정 '+partial.sides.length+'/4변 (전체 검증 아님)'):'';
const unstable=Object.keys(partial?.unstable_sides||{});const exposureText=unstable.length?(' · 밝기 변형에서 불안정한 경계 '+unstable.map(x=>edgeNames[x]||x).join('·')+' 제외'):'';
const edgeText=typeof edgeAudit.candidate_edge_count==='number'?' · 종이–바닥 광도 경계 후보 '+edgeAudit.candidate_edge_count+'/4'+(missing?' · 근거 부족: '+missing:''):'';
const shapeDecision=partial?.checker_shape_guard?.decision||'';const shapeText=shapeDecision==='reject_distorted_refit'?' · 검은 칸 형태 왜곡으로 A4 부분 보정 거부':shapeDecision==='no_strong_checker_contradiction'?' · 검은 칸 기하 악화 없음(외곽 검증 아님)':'';
const matching=data.same_board_ransac?.[f.kind]||{};
const rel=(matching.pairs||[]).find(p=>p.target===f.view_id);
const orbText=rel?.status==='relative_geometry_consistent'
  ? ` · L02/L05/L07 검은 칸 중심 RANSAC 상대 일치 (Inlier ${rel.inliers}개 · A4 외곽 검증 아님)`
  :rel?.status==='relative_geometry_consistent_limited'
    ? ' · L07 검은 칸 중심 상대 일치(대응점이 적어 제한적)'
    :rel?.status==='relative_geometry_disagreement'
    ? ' · L07 사진 간 사각형 상대 불일치 (추가 검증 필요)'
    : rel ? ' · L07 사진 간 특징점 대응 근거 부족'
    : matching.anchor===f.view_id ? ' · L07 사진 간 비교 기준(실제 A4 모서리 정답 아님)' : '';

diagnostic.textContent=(f.paper_method||'no A4')+(f.reference_transfer_evidence ? ` · 다른 사진 ${f.reference_transfer_evidence.reference}와의 검은 칸 RANSAC ${f.reference_transfer_evidence.inliers}점 검증(실제 A4 외곽 정답 아님)` : '')+(f.outer_border_verified?' · 종이 경계 증거 있음':' · A4 바깥 경계 검증 미완료')+' / 검은 칸 '+f.detected_squares+'개'+squareInfo+fitted+exposureText+edgeText+shapeText+orbText+(String(f.paper_method||'').startsWith('adaptive_')?' · 밝기 적응형 보조 검출':'')+(f.orientation_corrected?' · 촬영 방향 180° 자동 정렬':'');el.append(diagnostic);el.append(t);g.append(el)}
document.getElementById('result').style.display='block';document.getElementById('result').scrollIntoView({behavior:'smooth'});
}catch(e){alert(e.message)}finally{btn.disabled=false;btn.textContent='자동 분석 시작';}
});
</script></body></html>'''


def process_uploads(top_bytes,side_bytes,arch_bytes=None,*,top_names=None,side_names=None):
    def decode(raw):
        if not 100 < len(raw) <= MAX_IMAGE_BYTES:
            raise ValueError('개별 사진 파일이 비어 있거나 너무 큽니다')
        encoded=np.frombuffer(raw,np.uint8)
        # A reduced-resolution preflight makes oversized compressed images
        # fail before OpenCV allocates a potentially huge full-size bitmap.
        small=cv2.imdecode(encoded,cv2.IMREAD_REDUCED_COLOR_8)
        if small is None:
            raise ValueError('읽을 수 없는 이미지입니다')
        # Reduced decoding rounds dimensions; use a small tolerance to avoid
        # incorrectly rejecting a photograph near the boundary.
        estimate_h,estimate_w=small.shape[:2]
        estimate_pixels=estimate_h*estimate_w*64
        if (estimate_pixels > MAX_DECODED_IMAGE_PIXELS+60_000 or
                max(estimate_h,estimate_w)*8 > MAX_DECODED_SIDE+8):
            raise ValueError('이미지 해상도가 너무 큽니다. 장당 최대 약 2천만 픽셀입니다.')
        im=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
        if im is None or min(im.shape[:2])<120:
            raise ValueError('읽을 수 없는 이미지 또는 너무 낮은 해상도입니다')
        if max(im.shape[:2])>MAX_DECODED_SIDE or im.shape[0]*im.shape[1]>MAX_DECODED_IMAGE_PIXELS:
            raise ValueError('이미지 해상도가 너무 큽니다')
        # The estimator always resizes to 950px. Drop full-resolution buffers
        # NOW instead of keeping up to 16 smartphone bitmaps alive in RAM.
        return resize_image(im)
    if not (1<=len(top_bytes)<=MAX_PER_KIND and 1<=len(side_bytes)<=MAX_PER_KIND):
        raise ValueError('위쪽 및 측면 사진을 각각 1~8장 올려 주세요')
    if top_names is not None and len(top_names) != len(top_bytes):
        raise ValueError('위쪽 사진 파일 이름 수가 업로드 수와 다릅니다')
    if side_names is not None and len(side_names) != len(side_bytes):
        raise ValueError('측면 사진 파일 이름 수가 업로드 수와 다릅니다')
    top_images=[decode(v) for v in top_bytes]
    side_images=[decode(v) for v in side_bytes]
    items,res=analyze_batch(top_images,side_images)
    # L07 ORB / RANSAC checks relative paper alignment between photographs
    # of the SAME view category. This has no authority to modify millimetres.
    match_audit=audit_same_view_photos(top_images,side_images,items)
    def encode(image):
        if image is None:
            return None
        h,w=image.shape[:2]
        if max(h,w)>650:
            factor=650/max(h,w)
            image=cv2.resize(image,(round(w*factor),round(h*factor)),interpolation=cv2.INTER_AREA)
        ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,67])
        return base64.b64encode(jpeg.tobytes()).decode('ascii') if ok else None
    arch_result={'status': 'not_requested', 'class_name': None, 'confidence': None,
                 'model_unchanged': True, 'score_is_accuracy': False}
    if arch_bytes:
        arch_photo=decode(arch_bytes)
        try:
            prediction=predict_arch_checked(arch_photo)
            arch_result={k:v for k,v in prediction.items() if k!='preview_bgr'}
            arch_result['preview']=encode(prediction.get('preview_bgr'))
        except (RuntimeError, ImportError, OSError) as exc:
            # YOLO dependency and memory availability must not prevent the
            # lecture OpenCV foot dimensions from being returned.
            arch_result={'status':'unavailable','class_name':None,'confidence':None,
                         'error':str(exc), 'model_unchanged':True,
                         'score_is_accuracy':False, 'preview':None}
    frames=[]
    for f in items:
        record=f.json()
        names=top_names if f.kind=='top' else side_names
        if names is not None:
            index=int(f.view_id.split('-')[-1])-1
            # Client-provided names are just display labels; no disk access.
            record['source_filename']=Path(str(names[index]).replace('\\','/')).name[:120]
        record['overlay']=encode(f.image)
        record['rectified']=encode(f.rectified_image)
        record['mask_preview']=encode(f.mask_image)
        if f.rectified_image is not None and f.mask_image is not None:
            dimension=res['width' if f.kind=='top' else 'length']
            used=f.view_id in dimension['used_views']
            record['measurement_preview']=encode(annotated_measurement(
                f.rectified_image,f.mask_image,f.kind,accepted=f.accepted,
                used_in_fusion=used,millimeters=f.mm,
                reason=f.reason if not f.accepted else dimension['rejected'].get(f.view_id,f.reason)))
        else:
            record['measurement_preview']=None
        frames.append(record)
    side_frames=[f for f in frames if f['kind']=='side']
    side_report={'paper_candidate_found':sum(f['paper_corners'] is not None for f in side_frames),
                 'individual_length_measured':sum(f['mm'] is not None for f in side_frames),
                 'accepted_for_final':len(res['length']['used_views']),
                 'total':len(side_frames),
                 'warning':'A4 검출이 성공해도 종이 바깥선 오차 또는 사진 간 불일치로 최종 길이 계산에서 제외될 수 있습니다.'}
    return {'side_detection_summary':side_report, 'version':'footframe', 'board_profile':'existing_user_checkerboard_a4', 'requires_new_printed_template':False, 'frames':frames,'fusion':res,'same_board_ransac':match_audit, 'arch':arch_result,
            'disclaimer':'Lecture-based 2D projected estimates. Checkerboard bounding boxes are NOT physically calibrated printed patterns and do NOT verify outer A4 corners.'}


def process_arch_only(raw: bytes) -> dict:
    """A YOLO request must not depend on uploading top/side CV photos."""
    if not 100 < len(raw) <= MAX_IMAGE_BYTES:
        raise ValueError('발 아치 사진이 비어 있거나 용량 제한을 초과했습니다.')
    # Same reduced-decode preflight as the existing web path.
    encoded=np.frombuffer(raw, np.uint8)
    small=cv2.imdecode(encoded, cv2.IMREAD_REDUCED_COLOR_8)
    if small is None or min(small.shape[:2])<15:
        raise ValueError('이미지를 읽을 수 없습니다.')
    if (small.shape[0]*small.shape[1]*64 > MAX_DECODED_IMAGE_PIXELS+60_000
            or max(small.shape[:2])*8 > MAX_DECODED_SIDE+8):
        raise ValueError('이미지 해상도가 너무 큽니다.')
    full=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if full is None or min(full.shape[:2])<120:
        raise ValueError('유효한 아치 사진이 필요합니다.')
    image=resize_image(full)
    try:
        result=predict_arch_checked(image)
        preview=result.pop('preview_bgr',None)
        if preview is not None:
            h,w=preview.shape[:2]
            if max(h,w)>650:
                scale=650/max(h,w)
                preview=cv2.resize(preview,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
            ok,jpg=cv2.imencode('.jpg', preview,[cv2.IMWRITE_JPEG_QUALITY, 67])
            result['preview']=base64.b64encode(jpg.tobytes()).decode('ascii') if ok else None
        else:
            result['preview']=None
        return result
    except (RuntimeError,ImportError,OSError) as exc:
        return dict(status='unavailable', error=str(exc),class_name=None,
                    confidence=None, model_unchanged=True,
                    score_is_accuracy=False,preview=None,
                    diagnostic=arch_capability())


class RequestHandler(BaseHTTPRequestHandler):
    def send_data(self,status,content,ctype):
        body=content if isinstance(content,bytes) else content.encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self' data: 'unsafe-inline'; connect-src 'self'")
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path in ('/','/index.html'):
            self.send_data(200,HTML,'text/html; charset=utf-8')
        elif self.path=='/health':
            self.send_data(200,json.dumps({'status':'ok','version':'footframe',
                                            'arch':arch_capability()}),
                           'application/json; charset=utf-8')
        elif self.path=='/arch-status':
            self.send_data(200,json.dumps(arch_capability(),ensure_ascii=False),
                           'application/json; charset=utf-8')
        else:self.send_data(404,'Not found','text/plain')
    def do_POST(self):
        if self.path not in ('/analyze','/analyze-arch'):
            return self.send_data(404,'Not found','text/plain')
        if not _ANALYSIS_SLOT.acquire(blocking=False):
            return self.send_data(503,json.dumps({'error':'현재 다른 사진을 분석 중입니다. 잠시 후 다시 요청해 주세요.'},ensure_ascii=False),
                                  'application/json; charset=utf-8')
        try:
            if self.path=='/analyze-arch':
                self._process_arch_request()
            else:
                self._process_analyze_request()
        finally:
            _ANALYSIS_SLOT.release()

    def _process_arch_request(self):
        try:
            clen=int(self.headers.get('Content-Length','0'))
            if not 0<clen<MAX_IMAGE_BYTES+100_000:
                raise ValueError('업로드 용량 제한을 초과했습니다.')
            ctype=self.headers.get('Content-Type','')
            if 'multipart/form-data' not in ctype:
                raise ValueError('multipart/form-data 형식이 필요합니다.')
            message=BytesParser(policy=default).parsebytes(
                ('Content-Type: '+ctype+'\r\nMIME-Version: 1.0\r\n\r\n').encode('ascii')
                +self.rfile.read(clen))
            images=[part.get_payload(decode=True) or b'' for part in message.iter_parts()
                    if part.get_param('name',header='content-disposition')=='arch']
            if len(images)!=1:
                raise ValueError('YOLO 분석용 발 아치 사진 한 장을 선택해 주세요.')
            result=process_arch_only(images[0])
            status=503 if result.get('status')=='unavailable' else 200
            self.send_data(status,json.dumps({'arch':result},ensure_ascii=False),
                           'application/json; charset=utf-8')
        except (ValueError,cv2.error) as exc:
            self.send_data(400,json.dumps({'error':str(exc)},ensure_ascii=False),
                           'application/json; charset=utf-8')
        except Exception:
            traceback.print_exc()
            self.send_data(500,json.dumps({'error':'YOLO 실행 중 서버 내부 오류'},ensure_ascii=False),
                           'application/json; charset=utf-8')

    def _process_analyze_request(self):
        try:
            clen=int(self.headers.get('Content-Length','0'))
            if not 0<clen<MAX_BYTES:raise ValueError('업로드 용량은 총 34MB 미만이어야 합니다')
            ctype=self.headers.get('Content-Type','')
            if 'multipart/form-data' not in ctype:raise ValueError('multipart/form-data 형식이 필요합니다')
            raw=self.rfile.read(clen)
            header=('Content-Type: '+ctype+'\r\nMIME-Version: 1.0\r\n\r\n').encode('ascii')
            msg=BytesParser(policy=default).parsebytes(header+raw)
            parts={'top':[],'side':[], 'arch':[]}
            names={'top':[], 'side':[]}
            for part in msg.iter_parts():
                field=part.get_param('name',header='content-disposition')
                if field in parts:
                    if len(parts[field]) >= (MAX_ARCH_IMAGES if field == 'arch' else MAX_PER_KIND):raise ValueError('위쪽·측면 각각 최대 8장, 아치 사진 최대 1장입니다')
                    parts[field].append(part.get_payload(decode=True) or b'')
                    if field in names:
                        names[field].append(part.get_filename() or f'{field}-{len(parts[field])}')
            result=process_uploads(parts['top'],parts['side'],parts['arch'][0] if parts['arch'] else None,
                                   top_names=names['top'], side_names=names['side'])
            self.send_data(200,json.dumps(result,ensure_ascii=False,allow_nan=False),'application/json; charset=utf-8')
        except (ValueError,cv2.error) as exc:
            self.send_data(400,json.dumps({'error':str(exc)},ensure_ascii=False),'application/json; charset=utf-8')
        except Exception:
            traceback.print_exc()
            self.send_data(500,json.dumps({'error':'서버 내부 처리 오류가 발생했습니다. 서버 로그를 확인해 주세요.'},ensure_ascii=False),'application/json; charset=utf-8')

if __name__=='__main__':
    # Presentation-day diagnostic: actually run the YOLO checkpoint (if the
    # ultralytics environment is installed), instead of confusing file SHA256
    # verification with inference. This does not require starting a web server.
    import sys
    if '--test-yolo' in sys.argv:
        try:
            ix=sys.argv.index('--test-yolo')
            photo=Path(sys.argv[ix+1])
            image=cv2.imread(str(photo),cv2.IMREAD_COLOR)
            if image is None:raise ValueError(f'사진 파일을 읽을 수 없습니다: {photo}')
            status=arch_capability()
            print('YOLO 환경 확인:',json.dumps(status,ensure_ascii=False),flush=True)
            value=predict_arch_checked(resize_image(image))
            output=value.pop('preview_bgr',None)
            if output is not None:
                name=photo.with_name(photo.stem+'_yolo_result.jpg')
                if not cv2.imwrite(str(name),output):
                    raise RuntimeError('YOLO 결과 이미지를 저장하지 못했습니다.')
                value['result_image']=str(name)
            print('YOLO 실제 추론 결과:',json.dumps(value,ensure_ascii=False),flush=True)
        except Exception as exc:
            print('YOLO 실제 실행 실패:',str(exc),flush=True)
            print('설치: python -m pip install -r requirements-yolo.txt',flush=True)
            raise SystemExit(1)
        raise SystemExit(0)
    if '--check-yolo' in sys.argv:
        status=arch_capability()
        print(json.dumps(status,ensure_ascii=False,indent=2))
        raise SystemExit(0 if status['status']=='ready_to_attempt' else 1)
    host=os.getenv('HOST','0.0.0.0');port=int(os.getenv('PORT','8000'))
    print(f'FootFrame Lecture-CV + YOLO: http://{host}:{port}',flush=True)
    ThreadingHTTPServer((host,port),RequestHandler).serve_forever()
