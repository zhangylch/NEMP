subroutine inverse_matrix(matrix,inv_matrix)
    use constant
    implicit none
    real(kind=typenum) :: matrix(3,3),inv_matrix(3,3)
    real(kind=typenum) :: tmp(3,3)
    real(kind=typenum),allocatable :: work(:)
    real(kind=typenum) :: work_query(1) ! 用于 lwork 查询的临时数组
    integer(kind=intype) :: lwork_optimal, lwork
    integer(kind=intype) :: ipiv(3),info

    tmp=matrix
    ipiv=0
    call dgetrf(3,3,tmp,3,ipiv,info)

    lwork=-1
    call dgetri(3,tmp,3,ipiv,work_query,lwork,info) 
    lwork_optimal=int(work_query(1))

    allocate(work(lwork_optimal))
    call dgetri(3,tmp,3,ipiv,work,lwork_optimal,info)

    deallocate(work)
    inv_matrix=tmp
    return
 
end subroutine
