import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Please enter the shape of inner domain')
  shape = sys.argv[1]

  h_inv_bem = []
  fx_bem = []
  fy_bem = []
  with open(shape + '_bem.txt', 'r') as f_bem:
    start = False
    for line in f_bem:
      words = line.split()
      if not start:
        if words[0] == '1/h':
          start = True
      else:
        h_inv_bem.append(int(words[0]))
        fx_bem.append(float(words[1]))
        fy_bem.append(float(words[2]))

  h_inv_fem = []
  fx_fem = []
  fy_fem = []
  with open(shape + '_fem.txt', 'r') as f_fem:
    start = False
    for line in f_fem:
      words = line.split()
      if not start:
        if words[0] == '1/h':
          start = True
      else:
        h_inv_fem.append(int(words[0]))
        fx_fem.append(float(words[1]))
        fy_fem.append(float(words[2]))

  h_bem = 1. / np.array(h_inv_bem[:-1])
  exact = np.array([fx_bem[-1], fy_bem[-1]])
  data_bem = np.array([fx_bem[:-1],fy_bem[:-1]])
  error_bem = np.linalg.norm(data_bem - exact.reshape((2, 1)), axis=0)
  error_bem /= np.linalg.norm(exact)
  fit_bem = np.polyfit(np.log(h_bem), np.log(error_bem), 1)
  print(f'Pullback approach (BEM): algebraic convergence with rate {fit_bem[0]:.3}')

  last = 0
  for i in range(len(h_inv_bem)-1):
    if h_inv_fem[i] < h_inv_bem[-1]:
      last += 1
  h_fem = 1. / np.array(h_inv_fem[:last])
  data_fem = np.array([fx_fem[:last], fy_fem[:last]])
  error_fem = np.linalg.norm(data_fem - exact.reshape((2, 1)), axis=0)
  error_fem /= np.linalg.norm(exact)
  fit_fem = np.polyfit(np.log(h_fem), np.log(error_fem), 1)
  print(f'Volume formula (FEM): algebraic convergence with rate {fit_fem[0]:.3}')

  fig, ax = plt.subplots()
  ax.loglog(h_bem, h_bem**fit_bem[0] * np.exp(fit_bem[1]),
            color='silver', linestyle='--')
  ax.loglog(h_fem, h_fem**fit_fem[0] * np.exp(fit_fem[1]),
            color='silver', linestyle='--')
  ax.loglog(h_bem, error_bem, 'o-', label='Pullback approach (BEM)')
  ax.loglog(h_fem, error_fem, 'o-', label='Volume formula (FEM)')
  mid_bem = [h_bem[h_bem.size//2] * 2, error_bem[h_bem.size//2] / 2]
  ax.annotate(r'$\alpha$ = ' + f'{fit_bem[0]:.3}', xy=mid_bem)
  mid_fem = [h_fem[h_fem.size//2] / 2, error_fem[h_fem.size//2] * 2]
  ax.annotate(r'$\alpha$ = ' + f'{fit_fem[0]:.3}', xy=mid_fem)
  ax.set_xlabel('h')
  ax.set_ylabel('Relative error')
  ax.legend()
  plt.show()