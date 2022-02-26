class Grading_Bank_Average:
    
    def Final_Grade_Computation(self,lst):
        a=0
        Count=lst.count()
        for l in lst:
            if l=='Strong Buy':
                a=a+2
            elif l=='Outperform':
                a=a+2
            elif l=='Buy':
                a=a+1
            elif l=='Overweight':
                a=a+2
            elif l=='Sell':
                a=a-1
            elif l=='Hold':
                a=a-1
            elif l=='Underweight':
                a=a-2
            elif l=='Market Outperform':
                a=a+2
            elif l=='Positive':
                a=a+1
            else:
                a=a
                        
        grade=round(a/Count,2)
        #We use a degressig barema for the strong buy,making hard to have top notch grades          
        if grade<2 and grade>=1.5:
            a='Strong Buy Graded by Firms'
        elif grade<1.5 and grade>=0.5:
            a='Buy Graded by Firms'
        elif grade<0.5 and grade >=-0.5 :
            a='Hold Graded by Firms'
        elif grade>-1 and grade <=-0.5:
            a='Sell Graded by Firms'
        else:
            a='Strong Sell Graded by Firms'
        return a
        
    def __init__(self,lst):
        self.lst = lst
        self.Final_Grade_Given=self.Final_Grade_Computation(lst)
        
