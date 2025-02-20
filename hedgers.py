import pandas as pd
import math
from greeks import BS_delta, BS_vega, H_delta, H_vega
from greeks import BS_barrier, BS_digital, BS_GeometricAsian, BS_vanilla
from greeks import H_barrier, H_digital, H_GeometricAsian, H_vanilla
from calibrators import calibrate_BS, EuroCall, calibrate_H, H_EuroCall


def BS_D_Hedge(optiondata,optionprice):
    S = list(pd.read_csv("stocksim.csv")["S"]) #CSV containing simulated stock price
    strike = optiondata['Strike']
    maturity = optiondata['Maturity']/252
    prices = optiondata['Price']
    sigma = calibrate_BS([prices[0], maturity, strike], 100)
    delta = BS_delta(100,0,strike,maturity,sigma,optionprice)
    value = optionprice(100,strike,maturity,sigma)
    for i in range(1, math.ceil(252*maturity)):
        value -= delta*S[i-1]
        sigma = calibrate_BS([prices[i], maturity-(i/252), strike], S[i])
        delta = BS_delta(S[i],i/252,strike,maturity-(i/252),sigma, optionprice)
        value += delta*S[i]
    results = {"Strike": strike, "Maturity": optiondata["Maturity"], "XT": value}
    return results

def BS_DV_Hedge(optiondata, optionprice):
    S = list(pd.read_csv("stocksim.csv")["S"]) #CSV containing simulated stock price
    strike = optiondata['Strike']
    maturity = optiondata['Maturity']/252
    prices = optiondata['Price']
    sigma = calibrate_BS([prices[0], maturity, strike], 100)
    cash = prices[0]
    deltaV = BS_delta(100,0,strike,maturity,sigma,optionprice)
    vegaV = BS_vega(100,0,strike,maturity,sigma,optionprice)
    deltaU = (EuroCall(100+0.1,strike,maturity,0,sigma)-EuroCall(100,strike,maturity,0,sigma))/0.1
    vegaU = (EuroCall(100,strike,maturity,0,sigma+0.001)-EuroCall(100,strike,maturity,0,sigma))/0.001
    value = optionprice(100,strike,maturity,sigma)
    for i in range(1, math.ceil(252*maturity)):
        value -= (deltaV - deltaU*vegaV/vegaU)*S[i-1] + (vegaV/vegaU)*EuroCall(S[i-1],strike,maturity-(i/252),0,sigma)
        sigma = calibrate_BS([prices[i], maturity-(i/252), strike], S[i])
        deltaV = BS_delta(S[i],i/252,strike,maturity-(i/252),sigma, optionprice)
        vegaV = BS_vega(S[i],i/252,strike,maturity-(i/252),sigma,optionprice)
        deltaU = (EuroCall(S[i]+0.1,strike,maturity-(i/252),0,sigma)-EuroCall(S[i],strike,maturity-(i/252),0,sigma))/0.1
        vegaU = (EuroCall(S[i],strike,maturity-(i/252),0,sigma+0.001)-EuroCall(S[i],strike,maturity-(i/252),0,sigma))/0.001
        value += (deltaV - deltaU*vegaV/vegaU)*S[i] + (vegaV/vegaU)*EuroCall(S[i],strike,maturity,0,sigma)
    results = {"Strike": strike, "Maturity": optiondata["Maturity"], "XT": value}
    return results

def Heston_D_Hedge(optiondata,f,optionprice):
    S = list(pd.read_csv("stocksim.csv")["S"]) #CSV containing simulated stock price
    strike = optiondata['Strike']
    maturity = optiondata['Maturity']/252
    prices = optiondata['Price']
    z = calibrate_H([strike, f[0]["Strike"], f[1]["Strike"], f[2]["Strike"], f[3]["Strike"]], [maturity], [prices[0],f[0]["Price"][0],f[1]["Price"][0],f[2]["Price"][0],f[3]["Price"][0]], 100)
    sigma,kappa,theta,volvol,rho = z[3],z[0],z[1],z[2],z[4]
    delta = H_delta(100,strike,maturity,sigma,kappa,theta,volvol,rho,optionprice)
    value = optionprice(100,strike,maturity,sigma,kappa,theta,volvol,rho)
    for i in range(1, math.ceil(252*maturity)):
        value -= delta*S[i-1]
        z = calibrate_H([strike, f[0]["Strike"], f[1]["Strike"], f[2]["Strike"], f[3]["Strike"]], [maturity-(i/252)], [prices[i],f[0]["Price"][i],f[1]["Price"][i],f[2]["Price"][i],f[3]["Price"][i]], S[i])
        sigma,kappa,theta,volvol,rho = z[3],z[0],z[1],z[2],z[4]
        delta = H_delta(S[i],strike,maturity-(i/252),sigma,kappa,theta,volvol,rho,optionprice)
        value += delta*S[i]
    results = {"Strike": strike, "Maturity": optiondata["Maturity"], "XT": value}
    return results

def Heston_DV_Hedge(optiondata,f,optionprice):
    S = list(pd.read_csv("stocksim.csv")["S"]) #CSV containing simulated stock price
    strike = optiondata['Strike']
    maturity = optiondata['Maturity']/252
    prices = optiondata['Price']
    z = calibrate_H([strike, f[0]["Strike"], f[1]["Strike"], f[2]["Strike"], f[3]["Strike"]], [maturity], [prices[0],f[0]["Price"][0],f[1]["Price"][0],f[2]["Price"][0],f[3]["Price"][0]], 100)
    sigma,kappa,theta,volvol,rho = z[3],z[0],z[1],z[2],z[4]
    deltaV = H_delta(100,strike,maturity,sigma,kappa,theta,volvol,rho,optionprice)
    vegaV = H_vega(100,strike,maturity,sigma,kappa,theta,volvol,rho)
    deltaU = H_delta(100,strike,maturity,sigma,kappa,theta,volvol,rho,H_EuroCall)
    vegaU = H_vega(100,strike,maturity,sigma,kappa,theta,volvol,rho,H_EuroCall)
    value = optionprice(100,strike,maturity,sigma,kappa,theta,volvol,rho)
    for i in range(1, math.ceil(252*maturity)):
        value -= (deltaV - deltaU*vegaV/vegaU)*S[i-1] + (vegaV/vegaU)*H_EuroCall(S[i-1],strike,maturity-(i/252),sigma,kappa,theta,volvol,rho)
        z = calibrate_H([strike, f[0]["Strike"], f[1]["Strike"], f[2]["Strike"], f[3]["Strike"]], [maturity-(i/252)], [prices[i],f[0]["Price"][i],f[1]["Price"][i],f[2]["Price"][i],f[3]["Price"][i]], S[i])
        sigma,kappa,theta,volvol,rho = z[3],z[0],z[1],z[2],z[4]
        deltaV = H_delta(S[i],strike,maturity-(i/252),sigma,kappa,theta,volvol,rho,optionprice)
        vegaV = H_vega(S[i],strike,maturity-(i/252),sigma,kappa,theta,volvol,rho,optionprice)
        deltaU = H_delta(S[i],strike,maturity-(i/252),sigma,kappa,theta,volvol,rho,H_EuroCall)
        vegaU = H_vega(S[i],strike,maturity-(i/252),sigma,kappa,theta,volvol,rho,H_EuroCall)
        value += (deltaV - deltaU*vegaV/vegaU)*S[i] + (vegaV/vegaU)*H_EuroCall(S[i],strike,maturity-(i/252),sigma,kappa,theta,volvol,rho)
    results = {"Strike": strike, "Maturity": optiondata["Maturity"], "XT": value}
    return results

def vanilla_payoff(S,strike,maturity):
    return max(S[maturity]-strike,0)

def digital_payoff(S,strike,maturity):
    return int(S[maturity]>strike)

def asian_payoff(S,strike,maturity):
    A = math.exp((1/maturity)*sum([math.log(x) for x in S[:maturity]]))
    return max(A-strike,0)

def barrier_payoff(S,strike,maturity,B,type):
    if type == "di":
        if min(S)>B:
            return 0
        else:
            return max(S[maturity]-strike,0)
    if type == "do":
        if min(S)<=B:
            return 0
        else:
            return max(S[maturity]-strike,0)
    if type == "ui":
        if max(S)<B:
            return 0
        else:
            return max(S[maturity]-strike,0)
    if type == "uo":
        if max(S)>=B:
            return 0
        else:
            return max(S[maturity]-strike,0)

def BS_unhedged(payoff_function):
    S = list(pd.read_csv("stocksim.csv")["S"])
    if payoff_function != barrier_payoff:
        if payoff_function == vanilla_payoff:
            optionprice = BS_vanilla
        elif payoff_function == digital_payoff:
            optionprice = BS_digital
        elif payoff_function == asian_payoff:
            optionprice = BS_GeometricAsian
        strikes = [85,90,95,100,105,110,115]
        maturities = [25,63,126,252,504]
        for i in strikes:
            for j in maturities:
                print({"Strike": i, "Maturity": j, "XT": payoff_function(S,i,j) - optionprice(100,i,j,0,0.2)})
    elif payoff_function == barrier_payoff:
        strikes = [85,90,95,100,105,110,115]
        maturities = [25,63,126,252,504]
        Downs = [50,55,60,65,70,75,80]
        Ups = [120,125,130,135,140,145,150]
        for i in strikes:
            for j in maturities:
                for k in ["do", "di"]:
                    for B in Downs:
                        print({"Strike": i, "Maturity": j, "Type":k, "Barrier":B, "XT": payoff_function(S,i,j,B,k) - BS_barrier(100,i,j,0,0.2,B,k)})
                for k in ["uo","ui"]:
                    for B in Ups:
                        print({"Strike": i, "Maturity": j, "Type":k, "Barrier":B, "XT": payoff_function(S,i,j,B,k) - BS_barrier(100,i,j,0,0.2,B,k)})

def H_unhedged(payoff_function):
    S = list(pd.read_csv("stocksim.csv")["S"])
    if payoff_function != barrier_payoff:
        if payoff_function == vanilla_payoff:
            optionprice = H_vanilla
        elif payoff_function == digital_payoff:
            optionprice = H_digital
        elif payoff_function == asian_payoff:
            optionprice = H_GeometricAsian
        strikes = [85,90,95,100,105,110,115]
        maturities = [25,63,126,252,504]
        for i in strikes:
            for j in maturities:
                print({"Strike": i, "Maturity": j, "XT": payoff_function(S,i,j) - optionprice(100,i,j,0,0.2)})
    elif payoff_function == barrier_payoff:
        strikes = [85,90,95,100,105,110,115]
        maturities = [25,63,126,252,504]
        Downs = [50,55,60,65,70,75,80]
        Ups = [120,125,130,135,140,145,150]
        for i in strikes:
            for j in maturities:
                for k in ["do", "di"]:
                    for B in Downs:
                        print({"Strike": i, "Maturity": j, "Type":k, "Barrier":B, "XT": payoff_function(S,i,j,B,k) - H_barrier(100,i,j,0,0.2,B,k)})
                for k in ["uo","ui"]:
                    for B in Ups:
                        print({"Strike": i, "Maturity": j, "Type":k, "Barrier":B, "XT": payoff_function(S,i,j,B,k) - H_barrier(100,i,j,0,0.2,B,k)})