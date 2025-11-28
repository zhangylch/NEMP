subroutine inverse_matrix(matrix,inv_matrix)
    use constant
    implicit none
    real(kind=typenum) :: matrix(3,3),inv_matrix(3,3)
    real(kind=typenum) :: tmp(3,3)
    real(kind=typenum),allocatable :: work(:)
    real(kind=typenum) :: work_query(1) ! 用于 lwork 查询的临时数组
    integer(kind=intype) :: lwork_optimal, lwork
    integer(kind=intype) :: ipiv(3),info

    ! 步骤 1: 将矩阵复制到 tmp 并执行 LU 分解 (只执行一次)
    tmp=matrix
    ipiv=0
    call dgetrf(3,3,tmp,3,ipiv,info)

    ! (可选) 检查 dgetrf 是否成功
    ! if (info /= 0) then ...

    ! 步骤 2: 使用 lwork=-1 查询 dgetri 的最优工作空间大小
    lwork=-1
    ! dgetri 会将最优大小写入 work_query(1)
    call dgetri(3,tmp,3,ipiv,work_query,lwork,info) 
    lwork_optimal=int(work_query(1))

    ! 步骤 3: 分配正确大小的 work 数组并执行计算
    allocate(work(lwork_optimal))
    ! 使用 lwork_optimal 再次调用
    call dgetri(3,tmp,3,ipiv,work,lwork_optimal,info)

    ! (可选) 检查 dgetri 是否成功
    ! if (info /= 0) then ...

    ! 步骤 4: 释放内存并返回结果
    deallocate(work)
    inv_matrix=tmp
    return
 
end subroutine
