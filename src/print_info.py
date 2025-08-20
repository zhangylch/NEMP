import time

class Print_Info():
    def __init__(self, ferr):
        self.ferr = ferr
   
    def __call__(self, iepoch, lr, loss_train, loss_val, ploss_val):
        #output the error 
        self.ferr.write("Step= {:6},  lr= {:5e}  ".format(iepoch, lr))
        self.ferr.write("train: ")
        self.ferr.write("{:10e} ".format(loss_train))
        self.ferr.write("val: ")
        self.ferr.write("{:10e}  ".format(loss_val))
        self.ferr.write("val prop: ")
        for iprop in ploss_val:
            self.ferr.write("{:10e}  ".format(iprop))
            
        self.ferr.write(" \n")
        self.ferr.flush()
        
