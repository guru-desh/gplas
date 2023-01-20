import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
from sympy import *
from sympy.parsing.latex import parse_latex

st.set_page_config("GPLAS", layout="wide")
st.title("GPLAS")

num_odes = st.select_slider("Number of ODEs", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

variables = []
equations = []
for i in range(num_odes):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        d_dt = st.latex(r"\frac{d}{dt}(")
    with col2:
        var = st.text_input("Variable", value=f"X_{i}")
        variables.append(parse_latex(var))
    with col3:
        eq = st.latex(r") = ")
    with col4:
        equation = st.text_input("Enter ODE using Latex", value=f"X_{i}")
        equations.append(equation)
    
st.header("Parsed Input")
sympy_equations = []
for variable, equation in zip(variables, equations):
    st.latex(r"\frac{d}{dt}(" + f"{variable}) = {equation}")
    sympy_equations.append(parse_latex(equation))

# print(sympy_equations)

st.header("Initial Values")
initial_values = []
for i in range(num_odes):
    col1, col2 = st.columns(2, gap='small')
    with col1:
        iv = st.latex(f"{variables[i]}(0) = ")
    with col2:
        iv = st.number_input("Enter Initial Value", min_value=0.0, value=1.0, key=f"{variables[i]}_iv", step=0.000001)
        initial_values.append(iv)

st.header("Parameters")
parameters = []
for eqn in sympy_equations:
    for sym in eqn.free_symbols:
        if sym not in parameters:
            parameters.append(sym)

parameters = [str(param) for param in parameters if param not in variables]

param_values = []
for param in parameters:
    col1, col2 = st.columns(2, gap='small')
    with col1:
        st.latex(f"{param} = ")
    with col2:
        par = st.number_input("Enter Parameter Value", min_value=-1.797e+308, value=1.0, key=f"{param}_pv", step=0.000001)
        param_values.append(par)

st.header("Time")
start = st.number_input("Start Time", value=0.0, step=0.000001)
end = st.number_input("End Time", value=10.0, step=0.000001)
step = st.number_input("Step Size", value=0.1, step=0.000001)

value_range = np.arange(start, end, step)
param_dict = dict(zip(parameters, param_values))
for i in range(len(sympy_equations)):
    sympy_equations[i] = sympy_equations[i].subs(param_dict)

lambdify_eqns = [lambdify(variables, eqn) for eqn in sympy_equations]
def model(initial_values, t):
    curr_equations = []
    # for i in range(len(sympy_equations)):
    #     curr_equations.append(sympy_equations[i].subs(dict(zip(variables, initial_values))))
    # return [curr_equations[i].evalf() for i in range(len(curr_equations))]
    return [lambdify_eqns[i](*initial_values) for i in range(len(lambdify_eqns))]

solutions = odeint(model, initial_values, value_range)
sol_plot = go.Figure()
for i in range(len(variables)):
    sol_plot.add_trace(go.Scatter(x=value_range, y=solutions[:, i], name=variables[i].name))
st.plotly_chart(sol_plot, use_container_width=True)