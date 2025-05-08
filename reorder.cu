#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cmath>
#include <stdlib.h>

//定义LHS矩阵大小
#define M 4 //row
#define K 4 //column
#define LHS_SIZE M*K

//定义RHS矩阵大小
#define N 1//column
#define RHS_SIZE K*N

#define RESULT_SIZE M*N

//定义子矩阵大小
#define BLOCK_ROW_SIZE 2
#define BLOCK_COL_SIZE 2

typedef struct{
    int col_id;
    int num;
}COLUMN;

int cmp(const void *a,const void *b){
    return (*(COLUMN*)b).num-(*(COLUMN*)a).num;
}

//得到行重排数组
void get_row(int row[],float h_LHS[]){
    int row_index=0;
    int i,j,k;
    int row_flag[M]={0};
    for(i=0;i<M;i++){
        for(j=0;j<K;j++){
            if(fabs(h_LHS[i*M+j])>1e-6&&row_flag[i]==0){
                row[row_index++]=i;
                row_flag[i]=1;
                for(k=i;k<M;k++){
                    if(fabs(h_LHS[k*M+j])>1e-6&&row_flag[k]==0){
                        row[row_index++]=k;
                        row_flag[k]=1;
                    }
                    if(row_index==M)
                        break;
                }
            }
            if(row_index==M)
                break;
        }
    }
}

//得到列重排数组
void get_col(COLUMN h_col[],float h_LHS[],int row[]){
    int i,j;
    for(i=0;i<K;i++){
        for(j=0;j<M/BLOCK_ROW_SIZE;j++)
            h_col[j*K+i].col_id=i;
        for(j=0;j<M;j++)
            if(fabs(h_LHS[row[j]*M+i])>1e-6)
                h_col[j/BLOCK_ROW_SIZE*K+i].num++;
    }
    for(i=0;i<M/BLOCK_ROW_SIZE;i++)
        qsort(h_col+i*K,K,sizeof(COLUMN),cmp);
    
}

//对LHS矩阵进行行重排
__global__ void LHS_row_reorder(float *out,float *in,int *row){
    int row_id=blockIdx.y*M+threadIdx.y;
    int col_id=blockIdx.x*K+threadIdx.x;
    int index_out=row_id*M+col_id;
    int index_in=row[row_id]*M+col_id;
    out[index_out]=in[index_in];
}

//对LHS矩阵进行列重排
__global__ void LHS_col_reorder(float *out,float *in,COLUMN *col){
    int row_id=blockIdx.y*M+threadIdx.y;
    int c_id=blockIdx.x*K+threadIdx.x;
    int index_out=row_id*M+c_id;
    int index_in;
    index_in=row_id*M+col[row_id/BLOCK_ROW_SIZE*K+c_id].col_id;
    out[index_out]=in[index_in];
}

__global__ void get_nz_block(int *out,float *in){
    int row_id=blockIdx.y*M/BLOCK_ROW_SIZE+threadIdx.y;
    int col_id=blockIdx.x*K/BLOCK_COL_SIZE+threadIdx.x;
    int index_out=row_id*M/BLOCK_ROW_SIZE+col_id;
    int i,j;
    out[index_out]=0;
    for(i=0;i<BLOCK_ROW_SIZE;i++)
        for(j=0;j<BLOCK_COL_SIZE;j++)
            out[index_out]+=(fabs(in[(i+row_id*2)*M+j+col_id*2])>1e-6);
}


COLUMN h_col[(M/BLOCK_ROW_SIZE)*K];//列重排数组

int main(){
    // 创建一个 cuSPARSE 句柄
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    //在主机上创建并初始化LHS矩阵
    float h_LHS[]={1,0,0,0,0,2,0,0,3,0,1,0,0,4,0,0};

    //得到行重排数组
    int h_row[M]={0};
    get_row(h_row,h_LHS);

    //得到列重排数组
    get_col(h_col,h_LHS,h_row);

    //在GPU上声明变量
    float *d_LHS,*d_temp,*d_reordered;
    int *d_row;
    COLUMN *d_col;

    //申请GPU上的内存
    cudaMalloc((void**)&d_LHS,sizeof(float)*LHS_SIZE);
    cudaMalloc((void**)&d_temp,sizeof(float)*LHS_SIZE);
    cudaMalloc((void**)&d_reordered,sizeof(float)*LHS_SIZE);
    cudaMalloc((void**)&d_row,sizeof(int)*M);
    cudaMalloc((void**)&d_col,sizeof(COLUMN)*(M/BLOCK_ROW_SIZE)*K);

    //将数据复制到GPU
    cudaMemcpy(d_LHS,h_LHS,sizeof(float)*LHS_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(d_row,h_row,sizeof(int)*M,cudaMemcpyHostToDevice);
    cudaMemcpy(d_col,h_col,sizeof(COLUMN)*(M/BLOCK_ROW_SIZE)*K,cudaMemcpyHostToDevice);
    
    dim3 blockSize(M,K);
    dim3 gridSize(1,1);
    //对LHS矩阵进行行重排
    LHS_row_reorder<<<gridSize,blockSize>>>(d_temp,d_LHS,d_row);
    //对LHS矩阵进行列重排
    LHS_col_reorder<<<gridSize,blockSize>>>(d_reordered,d_temp,d_col);
    
    //得到非零块数量
    int *d_nz_block;
    cudaMalloc((void**)&d_nz_block,sizeof(int)*(M/BLOCK_ROW_SIZE)*(K/BLOCK_COL_SIZE));
    dim3 subBlockSize(M/BLOCK_ROW_SIZE,K/BLOCK_COL_SIZE);
    
    get_nz_block<<<gridSize,subBlockSize>>>(d_nz_block,d_reordered);

    //得到非零块数组
    int *h_nz_block=(int*)malloc(sizeof(int)*(M/BLOCK_ROW_SIZE)*(K/BLOCK_COL_SIZE));
    cudaMemcpy(h_nz_block,d_nz_block,sizeof(int)*(M/BLOCK_ROW_SIZE)*(K/BLOCK_COL_SIZE),cudaMemcpyDeviceToHost);
    float *h_reordered=(float*)malloc(sizeof(float)*LHS_SIZE);
    cudaMemcpy(h_reordered,d_reordered,sizeof(float)*LHS_SIZE,cudaMemcpyDeviceToHost);

    int nnzb=0;
    int i,j;

    int *h_bsrRowPtr=(int*)malloc(sizeof(int)*(M/BLOCK_ROW_SIZE+1));
    h_bsrRowPtr[0]=0;
    for(i=0;i<M/BLOCK_ROW_SIZE;i++){
        for(j=0;j<K/BLOCK_COL_SIZE;j++){
            nnzb+=(h_nz_block[i*M/BLOCK_ROW_SIZE+j]>0);
        }
        h_bsrRowPtr[i+1]=nnzb;
    }


    int *h_bsrColInd=(int*)malloc(sizeof(int)*nnzb);
    int index=0;
    for(i=0;i<M/BLOCK_ROW_SIZE;i++){
        for(j=0;j<K/BLOCK_COL_SIZE;j++){
            if(h_nz_block[i*M/BLOCK_ROW_SIZE+j]>0){
                h_bsrColInd[index++]=j;
            }
        }
    }

    index=0;
    float *h_bsrVal=(float*)malloc(sizeof(float)*nnzb*BLOCK_ROW_SIZE*BLOCK_COL_SIZE);
    for(i=0;i<M;i++)
        for(j=0;j<K;j++)
            if(h_nz_block[(i/BLOCK_ROW_SIZE)*M/BLOCK_ROW_SIZE+j/BLOCK_COL_SIZE]>0)
                h_bsrVal[index++]=h_reordered[i*M+j];

    float *d_bsrVal;
    int *d_bsrRowPtr,*d_bsrColInd;

    cudaMalloc((void**)&d_bsrVal,sizeof(float)*nnzb*BLOCK_ROW_SIZE*BLOCK_COL_SIZE);
    cudaMalloc((void**)&d_bsrRowPtr,sizeof(int)*(M/BLOCK_ROW_SIZE+1));
    cudaMalloc((void**)&d_bsrColInd,sizeof(int)*nnzb);

    cudaMemcpy(d_bsrVal,h_bsrVal,sizeof(float)*nnzb*BLOCK_ROW_SIZE*BLOCK_COL_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsrRowPtr,h_bsrRowPtr,sizeof(int)*(M/BLOCK_ROW_SIZE+1),cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsrColInd,h_bsrColInd,sizeof(int)*nnzb,cudaMemcpyHostToDevice);

    //初始化RHS矩阵(K*N)
    float h_RHS[]={0,2,1,4};
    float h_RHS1[RHS_SIZE];
    float h_RHS2[RHS_SIZE];
    for(i=0;i<K;i++){
        h_RHS1[i]=h_RHS[h_col[i].col_id];
    }
    std::cout<<std::endl;
    for(i=0;i<K;i++){
        h_RHS2[i]=h_RHS[h_col[K+i].col_id];
    }
    float *d_RHS,*d_RHS1,*d_RHS2,*d_result1,*d_result2;
    cudaMalloc((void**)&d_RHS,sizeof(float)*RHS_SIZE);
    cudaMalloc((void**)&d_RHS1,sizeof(float)*RHS_SIZE);
    cudaMalloc((void**)&d_RHS2,sizeof(float)*RHS_SIZE);
    cudaMalloc((void**)&d_result1,sizeof(float)*RESULT_SIZE);
    cudaMalloc((void**)&d_result2,sizeof(float)*RESULT_SIZE);
    cudaMemcpy(d_RHS,h_RHS,sizeof(float)*RHS_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(d_RHS1,h_RHS1,sizeof(float)*RHS_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(d_RHS2,h_RHS2,sizeof(float)*RHS_SIZE,cudaMemcpyHostToDevice);
    
    float alpha=1.0f;
    float beta=0.0f;
    cusparseMatDescr_t descr{nullptr};
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseSbsrmm(
        handle,
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        M/BLOCK_ROW_SIZE,
        N,
        K/BLOCK_COL_SIZE,
        nnzb,
        &alpha,
        descr,
        d_bsrVal,
        d_bsrRowPtr,
        d_bsrColInd,
        BLOCK_ROW_SIZE,
        d_RHS1,
        K,
        &beta,
        d_result1,
        M
    );

    cusparseSbsrmm(
        handle,
        CUSPARSE_DIRECTION_ROW,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        M/BLOCK_ROW_SIZE,
        N,
        K/BLOCK_COL_SIZE,
        nnzb,
        &alpha,
        descr,
        d_bsrVal,
        d_bsrRowPtr,
        d_bsrColInd,
        BLOCK_ROW_SIZE,
        d_RHS2,
        K,
        &beta,
        d_result2,
        M
    );

    float *h_result1=(float*)malloc(sizeof(float)*RESULT_SIZE);
    float *h_result2=(float*)malloc(sizeof(float)*RESULT_SIZE);
    cudaMemcpy(h_result1,d_result1,sizeof(float)*RESULT_SIZE,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result2,d_result2,sizeof(float)*RESULT_SIZE,cudaMemcpyDeviceToHost);

    float *h_result=(float*)malloc(sizeof(float)*RESULT_SIZE);
    for(i=0;i<M;i++){
        if(i<M/2){
            h_result[h_row[i]]=h_result1[i];
        }
        else{
            h_result[h_row[i]]=h_result2[i];
        }
    }
    //check
    for(i=0;i<M;i++){
        std::cout<<h_result[i]<<" ";
    }
    std::cout<<std::endl;
    //释放GPU内存
    cusparseDestroy(handle);

    cudaFree(d_LHS);
    cudaFree(d_temp);
    cudaFree(d_reordered);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_bsrColInd);
    cudaFree(d_bsrRowPtr);
    cudaFree(d_bsrVal);
    cudaFree(d_nz_block);
    cudaFree(d_RHS1);
    cudaFree(d_RHS2);
    cudaFree(d_RHS);
    cudaFree(d_result1);
    cudaFree(d_result2);

    return 0;
}
