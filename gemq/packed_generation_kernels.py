"""CUDA-only grouped FP16/BF16 lookup kernels; no weight-scale arithmetic."""
import torch
import triton
import triton.language as tl
@triton.jit
def kernel(X,Q,L,O,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 e=tl.program_id(0);pn=tl.program_id(1);pm=tl.program_id(2)
 start=0
 if e>0:start=tl.load(O+e-1)
 end=tl.load(O+e)
 rm=start+pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
 if start+pm*BM<end:
  acc=tl.zeros((BM,BN),tl.float32)
  for b in range(tl.cdiv(K,BK)):
   kk=b*BK+rk
   a=tl.load(X+rm[:,None]*K+kk[None,:],(rm[:,None]<end)&(kk[None,:]<K),0)
   q=tl.load(Q+e*(K//2)*N+(kk[:,None]//2)*N+rn[None,:],(kk[:,None]<K)&(rn[None,:]<N),0).to(tl.int32)
   code=(q>>((kk[:,None]%2)*4))&15
   w=tl.load(L+e*(K//128)*8*N+((kk[:,None]//128)*8+code)*N+rn[None,:],(kk[:,None]<K)&(rn[None,:]<N),0)
   acc=tl.dot(a,w,acc)
  tl.store(Y+rm[:,None]*N+rn[None,:],acc.to(Y.dtype.element_ty),(rm[:,None]<end)&(rn[None,:]<N))
def grouped_mm(x,q,lut,offs,*,max_rows_per_expert):
 config=(16,128,64,4) if q.shape[-1]>8192 else (16,64,64,4)
 M,K=x.shape;E,_,N=q.shape;bm,bn,bk,warps=config;y=torch.empty((M,N),dtype=x.dtype,device=x.device)
 kernel[(E,triton.cdiv(N,bn),triton.cdiv(max_rows_per_expert,bm))](x,q,lut,offs,y,M,N,K,bm,bn,bk,num_warps=warps,num_stages=3)
 return y
@triton.jit
def unpack_kernel(Q,L,W,N:tl.constexpr,K:tl.constexpr,TOTAL:tl.constexpr,B:tl.constexpr):
 idx=tl.program_id(0)*B+tl.arange(0,B);e=idx//(N*K);n=(idx//K)%N;k=idx%K
 q=tl.load(Q+e*(K//2)*N+(k//2)*N+n,idx<TOTAL,0).to(tl.int32);code=(q>>((k%2)*4))&15
 v=tl.load(L+e*(K//128)*8*N+((k//128)*8+code)*N+n,idx<TOTAL,0)
 tl.store(W+idx,v,idx<TOTAL)
def unpack_weight(q,lut,K):
 E,_,N=q.shape;w=torch.empty((E,N,K),dtype=lut.dtype,device=q.device)
 unpack_kernel[(triton.cdiv(w.numel(),1024),)](q,lut,w,N,K,w.numel(),1024)
 return w
