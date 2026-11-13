import dysts.flows as flow
from dysts.flows import *
import dysts.systems
from sindy_exp._dysts_to_sympy import dynsys_to_sympy
from logging import getLogger

import dysts.flows
import dysts.systems
import numpy as np
import sympy as sp

from ODEClass import ODE
import csv
maxDegree = 2
ODE_CLASSES = {
    klass.lower(): getattr(dysts.flows, klass)
    for klass in dysts.systems.get_attractor_list()
}
#Filter to 3D Systems
DATA = dysts.systems.get_system_data()
ThreeDimODEs = {}
for item in DATA:
    if(DATA[item]['embedding_dimension']==3):
        ThreeDimODEs[item.lower()] = ODE_CLASSES[item.lower()]
expressions = {}
listOfErrors = []
symbolList = []
errorCount = 0
for item in ThreeDimODEs:
    try:
        tmp = ThreeDimODEs[item]()
        # print(item)
        # print(tmp)
        symbols, exprs, lambda_rhs = dynsys_to_sympy(tmp)
        #Filter out nonpolynomials and polynomials of degree>maxDegree
        if not (symbols==[sp.Symbol('x'),sp.Symbol('y'),sp.Symbol('z')]):
            raise Exception()
        for expr in exprs:
            if not (expr.is_polynomial()):
                raise Exception()
            expr.expand()
            degList = sp.Poly(expr).degree_list()
            if(max(degList)>maxDegree):
                raise Exception()
        #only our focus remains
        expressions[item] = exprs
    except Exception:
        errorCount+=1
        listOfErrors.append(item)  
testExpr = sp.Poly(expressions['lorenz'][0],sp.Symbol('x'),sp.Symbol('y'),sp.Symbol('z'))
def getIndexFromKey(key):
    strKey = str(key)
    strList = strKey.split(',')
    index = 0
    for i in range(3):
        curStr = strList[i]
        num = ''
        for char in curStr:
            if char.isnumeric():
                num+=char
        index += (int(num)*(3**(2-i)))
    return index
def expressionsToMatrix(exprList):
    W = np.zeros((3,27))
    for i in range(3):
        curExpr = exprList[i]
        poly = sp.Poly(curExpr,sp.Symbol('x'),sp.Symbol('y'),sp.Symbol('z'))
        exprDict = poly.as_dict()
        for key in exprDict:
            index = getIndexFromKey(key)
            W[i,index] = exprDict[key]
    return W
with open('attractorDataset.csv',"w") as file:
    header = "Name,1,z,z²,y,yz,yz²,y²,y²z,y²z²,x,xz,xz²,xy,xyz,xyz²,xy²,xy²z,xy²z²,x²,x²z,x²z²,x²y,x²yz,x²yz²,x²y²,x²y²z,x²y²z²,1,z,z²,y,yz,yz²,y²,y²z,y²z²,x,xz,xz²,xy,xyz,xyz²,xy²,xy²z,xy²z²,x²,x²z,x²z²,x²y,x²yz,x²yz²,x²y²,x²y²z,x²y²z²,1,z,z²,y,yz,yz²,y²,y²z,y²z²,x,xz,xz²,xy,xyz,xyz²,xy²,xy²z,xy²z²,x²,x²z,x²z²,x²y,x²yz,x²yz²,x²y²,x²y²z,x²y²z²"
    print(header,file=file)
    for key in expressions:
        curStr = f"{key}"
        flattenedW = expressionsToMatrix(expressions[key]).flatten()
        for num in flattenedW:
            curStr+=f",{num}"
        print(curStr,file=file)
    file.close()