import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import trange
import scipy.stats as stats
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

device = ("cuda" if torch.cuda.is_available() else "cpu")

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout(model, x_unknown, iterations, tasks):
  list_2 = np.array([])
  for iteration in range(iterations):
    enable_dropout(model)
    predictions = model(x_unknown.type(torch.float32).reshape(-1,1))
    if len(list_2) == 0:
      list_2 = np.array(predictions.cpu().detach().numpy().reshape(-1,x_unknown.shape[0], tasks))
    else:
      list_2 = np.concatenate((list_2, np.array(predictions.cpu().detach().numpy()).reshape(-1,x_unknown.shape[0], tasks)), axis=-0)

  mean = np.mean(list_2, axis = 0)
  std = np.std(list_2, axis = 0)
  return mean, std

def pi(model, x, iterations, tasks):
  mean, std = mc_dropout(model, x, iterations, tasks)
  dist = stats.norm(mean, std)
  max = np.max(mean)
  pi = 1 - dist.cdf(max)
  return pi

def ei(model, x, iterations, tasks, xi = 0.001):
  mean, std = mc_dropout(model, x, iterations, tasks)
  dist = stats.norm(mean, std)
  max = np.max(mean)
  try:
    z = (max + xi - mean) / std
    E = (mean - max - xi) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
  except:
    z = 1
    E = 100
  return E

def multi_task_aq_pi(model, x, tasks, iterations = 10, samples = 10):
  p = np.array([])
  for i in range(tasks):
    if i != 0:
      p *= pi(model, x, iterations, tasks)[:,i]
    else:
      p = pi(model, x, iterations, tasks)[:,i]
  return np.argsort(p, axis = 0)[-samples:]

def multi_task_aq_random(x, samples = 10):
  idx = np.random.choice(x.shape[0]-1, samples)
  return idx

def multi_task_aq_avg_std(model, x, tasks, iterations = 10, samples = 10):
  _, std = mc_dropout(model, x, iterations, tasks)
  return np.argsort(np.sum(std, axis = 1), axis = 0)[-samples:], np.sum(std, axis = 1)

def multi_task_aq_prod_std(model, x, tasks, iterations = 10, samples = 10):
  _, std = mc_dropout(model, x, iterations, tasks)
  return np.argsort(np.prod(std, axis = 1), axis = 0)[-samples: ], np.prod(std, axis = 1)

def multi_task_aq_ei(model, x, tasks, iterations = 10, samples = 10):
  p = np.array([])
  for i in range(tasks):
    if i != 0:
      p += ei(model, x, iterations, tasks)[:,i]
    else:
      p = ei(model, x, iterations, tasks)[:,i]
  return np.argsort(p, axis = 0)[-samples:]

def ranking(model, x, tasks, iterations = 10, samples = 10):
  _, std = mc_dropout(model, x, iterations, tasks)
  a = np.argsort(np.sum(np.argsort(std, axis = 0), axis = 1), axis = 0)
  return a[-samples:]

def round_robin(model, x, tasks, column, iterations = 10, samples = 10):
  _, std = mc_dropout(model, x, iterations, tasks)
  return np.argsort(std, axis = 0)[:, column][-samples: ]

def active_learning(x, y, known, epochs, samples, mc_iterations, task_num, heuristic, mybar, initdisplay = True):
  torch.manual_seed(42)
  np.random.seed(42)

  idx = np.random.choice(x.shape[0], known)
  x_known, y_known = x[idx], y[idx]
  
  if initdisplay:
    st.header("Datasets")
    fig = plt.figure(figsize = (10,7))
    for i in range(task_num):
      ax = fig.add_subplot(2, task_num, i+1)
      ax.plot(x.cpu().detach(), y[:,i].cpu().detach(), color = "blue")
    plt.show()
    st.pyplot()
    
    st.header("Initial Points")
    fig = plt.figure(figsize = (10,7))
    for i in range(task_num):
      ax = fig.add_subplot(2, task_num, i+1)
      ax.scatter(x_known.cpu().detach(), y_known[:,i].cpu().detach(), marker = "o", color = "red")
    plt.show()
    st.pyplot()

  mask1 = torch.full(x.shape, True)
  mask1[idx] = False
  x_unknown = torch.masked_select(x, mask1.to(device))
  mask2 = torch.full(y.shape, True)
  mask2[idx] = False
  y_unknown = torch.masked_select(y, mask2.to(device)).reshape(-1, (y_known.shape[1]))

  model = regression_non_linear(1, task_num).to(device)
  opt = torch.optim.Adam(model.parameters(), lr = 1e-2)
  lossfn = MSELoss()
  
  fig = plt.figure(figsize = (10,7))
  iterations = 100
  
  for epoch in trange(epochs):
    for iteration in range(iterations):
      model.train()
      pred = model(x_known.reshape(-1,1))
      loss = lossfn(pred, y_known)
      loss.backward()
      opt.step()
      opt.zero_grad()

    if heuristic == "avg_std":
      idx, std = multi_task_aq_avg_std(model, x_unknown, task_num, mc_iterations, samples)
    elif heuristic == "prod_std":
      idx, std = multi_task_aq_prod_std(model, x_unknown, task_num, mc_iterations, samples)
    elif heuristic == "random":
      idx = multi_task_aq_random(x_unknown, samples)
    elif heuristic == "pi":
      idx = multi_task_aq_pi(model, x_unknown, task_num, mc_iterations, samples)
    elif heuristic == "ei":
      idx = multi_task_aq_ei(model, x_unknown, task_num, mc_iterations, samples)
    elif heuristic == "ranking":
      idx = ranking(model, x_unknown, task_num, mc_iterations, samples)
    elif heuristic == "round_robin":
      idx = round_robin(model, x_unknown, task_num, epoch%task_num, mc_iterations, samples)
    else:
      print("Choose Correct Heuristic. Legal heuristics: [`pi`, `ei`, `random`, `prod_std`, `avg_std`, `ranking`, `round_robin`]")
      break
    
    mask1 = torch.full(x_unknown.shape, False)
    mask1[idx] = True
    x_select = torch.masked_select(x_unknown, mask1.to(device))
    mask2 = torch.full(y_unknown.shape, False)
    mask2[idx] = True
    y_select = torch.masked_select(y_unknown, mask2.to(device)).reshape(-1, (y_known.shape[1]))

    x_known = torch.cat((x_known, x_select), axis=0)
    y_known = torch.cat((y_known, y_select), axis=0)

    mask1 = torch.full(x_unknown.shape, True)
    mask1[idx] = False
    x_unknown = torch.masked_select(x_unknown, mask1.to(device))
    mask2 = torch.full(y_unknown.shape, True)
    mask2[idx] = False
    y_unknown = torch.masked_select(y_unknown, mask2.to(device)).reshape(-1, (y_known.shape[1]))

    if epoch%5 == 0:
      print("Loss: {}".format(loss))

    if int((epoch+1)*100/epochs)<80:
      mybar.progress(int((epoch+1)*50/epochs + int(not initdisplay)*50), text = "Learning...")
    else:
      if int((epoch+1)*100/epochs)<100:
        mybar.progress(int((epoch+1)*50/epochs + int(not initdisplay)*50), text = "Almost There...")
      else:
        mybar.progress(int((epoch+1)*50/epochs + int(not initdisplay)*50), text = "Done!")
  
  st.header(heuristic)

  model.eval()
  fig = plt.figure(figsize = (10,7))
  for i in range(task_num):
    ax = fig.add_subplot(2, task_num, i+1)

    pred = torch.tensor([])
    for j in range(100):
      enable_dropout(model)
      if j==0:
        pred = model(x.reshape(-1,1))[:,i].cpu().detach().reshape(-1, model(x.reshape(-1,1))[:,i].cpu().detach().shape[0])
      else:
        pred = torch.cat((pred, model(x.reshape(-1,1))[:,i].cpu().detach().reshape(-1, model(x.reshape(-1,1))[:,i].cpu().detach().shape[0])), dim = 0)
    mean_pred = torch.mean(pred, axis = 0)
    std_pred = torch.std(pred, axis = 0)

    ax.scatter(x_known.cpu().detach(), y_known[:,i].cpu().detach(), marker = "x", color = "green", label = "Known")
    ax.plot(x.cpu().detach(), mean_pred, color = "blue", label = "Predictions")
    ax.fill_between(x.cpu().detach(), mean_pred - std_pred, mean_pred + std_pred, color = "gray", alpha = 0.5, label = "Uncertainty")
    ax.plot(x.cpu().detach(), y[:,i].cpu().detach(), color = "red", label = "True Plot")
    loss = lossfn(mean_pred, y[:, i].cpu().detach())
    ax.set_title("Loss: {}".format(round(float(loss),4)))
    ax.legend(loc = "best")

  plt.show()
  st.pyplot()
  st.code("Final loss: {}".format(lossfn(model(x.reshape(-1,1)), y)), language = "python")

class regression_non_linear(nn.Module):
  def __init__(self, input_features, output_features, hidden_units = 8):
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.GELU(),
        nn.Linear(hidden_units, hidden_units),
        nn.GELU(),
        nn.Linear(hidden_units, hidden_units),
        nn.Sigmoid(),
        nn.Dropout(p = 0.1),
        nn.Linear(hidden_units, hidden_units),
        nn.GELU(),
        nn.Linear(hidden_units, hidden_units),
        nn.GELU(),
        nn.Linear(hidden_units, output_features)
    )
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.linear_layer_stack(x)
  
st.markdown("<h1 style = 'text-align: center;'> MultiTask </h1>", unsafe_allow_html = True)

x = torch.linspace(-3, 3, 300).to(device)
y1 = 10*x
y1 = y1.reshape(-1,1)
y2 = x**2
y2 = y2.reshape(-1,1)
y3 = x**3
y3 = y3.reshape(-1,1)
y4 = torch.exp(x)
y4 = y4.reshape(-1,1)
y5 = torch.sin(5*x)
y5 = y5.reshape(-1,1)

list_ = []

options = ["linear", "parabola", "cube", "exponential", "sine"]
tasks = st.multiselect("Select Tasks: ", options, max_selections = 5)

if (options[0] in tasks):
    list_.append(y1)
if (options[1] in tasks):
    list_.append(y2)
if (options[2] in tasks):
    list_.append(y3)
if (options[3] in tasks):
    list_.append(y4)
if (options[4] in tasks):
    list_.append(y5)

task_num = len(list_)

y = torch.tensor([]).to(device)
for val in list_:
  y = torch.cat((y, val), axis = 1)

heuristics = ["pi", "ei", "prod_std", "avg_std", "ranking", "round_robin"]
heuristic = st.radio('Select a heuristic:', heuristics)

epochs = st.slider("Choose Iterations: ", 0, 300)

if "play" not in st.session_state:
  st.session_state["play"] = False

play = st.button("Play", use_container_width = True, type = "primary")

if play:
  st.session_state["play"] = True

if st.session_state["play"]:
  mybar = st.progress(0, text = "Learning...")
  active_learning(x, y, 10, epochs = epochs, samples = 1, mc_iterations = 2000, task_num = task_num, heuristic = heuristic, mybar = mybar)
  active_learning(x, y, 10, epochs = epochs, samples = 1, mc_iterations = 2000, task_num = task_num, heuristic = "random", mybar = mybar, initdisplay = False)
  st.session_state["play"] = False