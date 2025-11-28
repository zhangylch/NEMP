subroutine get_neigh(cart, coor, atomindex, shifts, maxneigh, numatom, scutnum)
     use constant
     use initmod
     implicit none
     integer(kind=intype),intent(in) :: maxneigh, numatom
     integer(kind=intype),intent(out) :: atomindex(2,maxneigh)
     integer(kind=intype),intent(out) :: scutnum
     integer(kind=intype) :: num, iatom, ninit, i, j, k, l, i1, i2, i3
     integer(kind=intype) :: sca(3), boundary(2,3), rangebox(3)
     integer(kind=intype),allocatable :: index_numrs(:,:,:,:,:)
     integer(kind=intype),allocatable :: index_rs(:,:,:)
     integer(kind=intype) :: max_neigh_peratom, thread_start_idx
       
     real(kind=typenum) :: tmp
     real(kind=typenum),intent(in) :: cart(3,numatom)
     real(kind=typenum),intent(out) :: shifts(3,maxneigh)
     real(kind=typenum),intent(out) :: coor(3,numatom)
     real(kind=typenum) :: oriminv(3), orimaxv(3), tmp1(3), rangecoor(3)
     real(kind=typenum) :: fcoor(3,numatom), imageatom(3,numatom,length), shiftvalue(3,length)
     
     integer(kind=intype) :: local_idx
     integer(kind=intype) :: local_scutnum(numatom)
     integer(kind=intype), allocatable :: local_atomindex(:,:,:) 
     real(kind=typenum), allocatable :: local_shifts(:,:,:)
     ! --------------------

       coor=cart
       fcoor=matmul(inv_matrix,coor)
!move all atoms to an cell ...
       oriminv=coor(:,1)
       orimaxv=coor(:,1)
       do iatom=2,numatom
         sca = nint(fcoor(:,iatom) - fcoor(:,1))
         coor(:,iatom) = coor(:,iatom) - sca(1)*matrix(:,1) - sca(2)*matrix(:,2) - sca(3)*matrix(:,3)
         do j=1,3
           if(coor(j,iatom)<oriminv(j)) then
             oriminv(j)=coor(j,iatom)
           else if(coor(j,iatom)>orimaxv(j)) then
             orimaxv(j)=coor(j,iatom)
           end if
         end do
       end do
       rangecoor=orimaxv-oriminv+2.0*rc
       rangebox=ceiling(rangecoor/dier)
       oriminv=oriminv-rc
       do iatom=1,numatom
         coor(:,iatom)=coor(:,iatom)-oriminv
       end do
       max_neigh_peratom = int((maxneigh / numatom) * capacity)
       allocate(index_rs(rangebox(1),rangebox(2),rangebox(3)))
       allocate(index_numrs(2, max_neigh_peratom, rangebox(1),rangebox(2),rangebox(3)))
       index_rs=0
!obtain image 
       l=0
       !$OMP PARALLEL DO PRIVATE(i, j, k, l, tmp1, iatom, sca, local_idx) &
       !$OMP SHARED(nimage, shiftvalue, matrix, numatom, imageatom, coor, rangecoor, dier, index_rs, index_numrs) &
       !$OMP COLLAPSE(3)
       do i=-nimage(3),nimage(3)
         do j=-nimage(2),nimage(2)
           do k=-nimage(1),nimage(1)
             l = (i+nimage(3)) * (2*nimage(2)+1)*(2*nimage(1)+1) + &
                 (j+nimage(2)) * (2*nimage(1)+1) + &
                 (k+nimage(1)) + 1
             shiftvalue(1,l)=k
             shiftvalue(2,l)=j
             shiftvalue(3,l)=i
             tmp1=matmul(matrix,shiftvalue(:,l))
             do iatom=1,numatom
               imageatom(:,iatom,l)=coor(:,iatom)+tmp1
               if(imageatom(1,iatom,l)>0d0 .and. imageatom(1,iatom,l)<rangecoor(1).and. &
               imageatom(2,iatom,l)>0d0 .and. imageatom(2,iatom,l)<rangecoor(2)  &
               .and. imageatom(3,iatom,l)>0d0 .and. imageatom(3,iatom,l)<rangecoor(3)) then
                 sca=ceiling(imageatom(:,iatom,l)/dier)
                 !$OMP ATOMIC CAPTURE
                 index_rs(sca(1),sca(2),sca(3))=index_rs(sca(1),sca(2),sca(3))+1
                 local_idx = index_rs(sca(1),sca(2),sca(3))
                 !$OMP END ATOMIC
                 index_numrs(:,local_idx,sca(1),sca(2),sca(3))=[iatom,l] 
               end if
             end do
           end do
         end do
       end do
       !$OMP END PARALLEL DO

       ninit=(length+1)/2

       allocate(local_atomindex(2, max_neigh_peratom, numatom))
       allocate(local_shifts(3, max_neigh_peratom, numatom))
       local_scutnum = 0

       !$OMP PARALLEL DO PRIVATE(iatom, sca, boundary, i, i1, i2, i3, j, l, tmp1, tmp) &
       !$OMP SHARED(numatom, coor, dier, ninit, imageatom, rangebox, interaction, &
       !$OMP index_rs, index_numrs, rcsq, shiftvalue, &
       !$OMP local_scutnum, local_atomindex, local_shifts, max_neigh_peratom)
       do iatom = 1, numatom
         sca=ceiling(coor(:,iatom)/dier)
  
         do i=1,3
           boundary(1,i)=max(1,sca(i)-interaction)
           boundary(2,i)=min(rangebox(i), sca(i)+interaction)
         end do
         do i3=boundary(1,3),boundary(2,3)
           do i2=boundary(1,2),boundary(2,2)
             do i1=boundary(1,1),boundary(2,1)
               do i=1,index_rs(i1,i2,i3)
                 j=index_numrs(1,i,i1,i2,i3)
                 l=index_numrs(2,i,i1,i2,i3)
                 tmp1 = imageatom(:,j,l) - coor(:,iatom)
                 tmp = dot_product(tmp1, tmp1)
                 if(tmp<=rcsq .and. tmp>0.0001) then
                   local_scutnum(iatom) = local_scutnum(iatom) + 1
                   local_atomindex(:, local_scutnum(iatom), iatom) = [iatom-1, j-1]
                   local_shifts(:, local_scutnum(iatom), iatom) = shiftvalue(:,l)
                 end if
               end do
             end do
           end do
         end do
       end do
       !$OMP END PARALLEL DO

       scutnum = 0
       do iatom = 1, numatom
         if (local_scutnum(iatom) < max_neigh_peratom + 0.5) then
           thread_start_idx = scutnum + 1
           scutnum = scutnum + local_scutnum(iatom)
           
           ! 检查总列表是否溢出
           if (scutnum > maxneigh) then
             print *, "ERROR: Total neighbor list overflow. Increase maxneigh."
             goto 999 
           end if
           
           atomindex(:, thread_start_idx : scutnum) = local_atomindex(:, 1 : local_scutnum(iatom), iatom)
           shifts(:, thread_start_idx : scutnum) = local_shifts(:, 1 : local_scutnum(iatom), iatom)
         
         else if (local_scutnum(iatom) > max_neigh_peratom + 0.5) then
           ! 处理单个原子溢出的错误
           print *, "ERROR: Per-atom neighbor list overflow for atom ", iatom
           print *, "Increase times_neigh."
           goto 999
         end if
       end do
 999   continue 

       deallocate(local_atomindex)
       deallocate(local_shifts)
       deallocate(index_numrs)
       deallocate(index_rs)
       atomindex(:,scutnum+1:maxneigh)=0
       shifts(:, scutnum+1:maxneigh)=0.0
     return
end subroutine get_neigh
