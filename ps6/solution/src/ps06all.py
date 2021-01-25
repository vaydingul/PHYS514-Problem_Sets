import ps0618, ps0619
ps = [ps0618, ps0619]
soln = [18, 19]
# ! PLEASE CLOSE FIGURES AFTER OBSERVING THEM. 
# ! IN THIS WAY, OTHER FIGURES IN OTHER TEST SCRIPTS CAN BE DISPLAYED


# ! ALSO, PS0619.PY TAKES A LITTLE BIT LONGER TO
# ! EXECUTE, BECAUSE, THEY HAVE INCLUDED ANAYLYSIS OF 
# ! ELAPSED TIME OF DIFFERENT METHODS.
# ! RIGHT NOW, THE TIME CALCULATIONS ARE BASED ON 1 ITERATION TO DEMONSTRATE
# ! THE CODE, IN THE REPORT, HIGHER NUMBERS ARE USED.
if __name__ == "__main__":

    for (ix,p) in enumerate(ps):

        print("======================Soluton of Question {0}=============================".format(soln[ix]))
        p.mytests()