module constant
     implicit none
     integer(kind=4), parameter :: intype=4, typenum=8
end module
module initmod
     use constant
     implicit none
     integer(kind=intype) :: interaction, length
     integer(kind=intype) :: nimage(3), pbc(3)
     real(kind=typenum) :: rc, rcsq, coeff
     real(kind=typenum) :: matrix(3, 3), inv_matrix(3, 3)
     real(kind=typenum) :: dier ! "dier" is the side length of the box used in cell-linked 
end module

subroutine init_neigh(in_rc, in_dier, cell, pbc_tmp)
     use constant
     use initmod
     implicit none
     real(kind=typenum),intent(in) :: in_rc, in_dier, cell(3,3)
     integer(kind=intype), intent(in) :: pbc_tmp(3)
     real(kind=typenum) :: dface(3), V
       rc=in_rc
       rcsq=rc*rc
       matrix=cell
       pbc = pbc_tmp
!Note that the fortran store the array with the column first, so the lattice parameters is the transpose of the its realistic shape
       call calculate_face_distances(cell, dface, V)
       dier=in_dier + 0.0001
       interaction=ceiling(rc/dier)
       nimage=ceiling(rc/abs(dface)) * pbc
       length=(2*nimage(1)+1)*(2*nimage(2)+1)*(2*nimage(3)+1)
       call inverse_matrix(matrix,inv_matrix)

     return
end subroutine

subroutine cross(a, b, c) 
    use constant
    implicit none
    real(kind=typenum) :: a(3), b(3)
    real(kind=typenum) :: c(3)
    c(1) = a(2) * b(3) - a(3) * b(2)
    c(2) = a(3) * b(1) - a(1) * b(3)
    c(3) = a(1) * b(2) - a(2) * b(1)
    return
end subroutine

subroutine norm(a, dist) 
    use constant
    implicit none
    real(kind=typenum) :: a(3)
    real(kind=typenum) :: dist
    dist = dsqrt(a(1)**2 + a(2)**2 + a(3)**2)
    return
end subroutine

subroutine calculate_face_distances(cell, dface, V)
    use constant
    implicit none
    real(kind=typenum) :: cell(3, 3)
    real(kind=typenum) :: dface(3), dist
    real(kind=typenum) :: area(3), a(3), b(3), c(3), a1(3), b1(3), c1(3), V

    a = cell(:, 1)
    b = cell(:, 2)
    c = cell(:, 3)

    call cross(b, c, a1)
    V = a(1)*a1(1) + a(2)*a1(2) + a(3)*a1(3)
    call cross(a, b, c1)
    call cross(c, a, b1)
    call norm(a1, dist)
    dface(1) = V / dist
    call norm(b1, dist)
    dface(2) = V / dist
    call norm(c1, dist)
    dface(3) = V / dist

    return
end subroutine 
