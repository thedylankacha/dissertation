# dissertation
Simulation code and data for my dissertation

Dissertation Title: CComparing hedging performance of standard volatility models under severe model risk

calcs.py - contains the necessary outputs: Greeks, Options Prices, Simulated Stock and Volatility paths
calibrators.py - contains the fitting functions for the Heston and Black-Scholes models to the sim data
hedgers.py - reads a csv of simdata and uses the calibrators to calculate the performance of a portfolio: unhedged, delta-hedged and vega-delta-hedged
