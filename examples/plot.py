import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('Please enter the shape of inner domain and epsilon2')
  shape = sys.argv[1]
  eps_2 = sys.argv[2]

  h_inv_bem = []
  fx_bem = []
  fy_bem = []
  fx_bem_bdry = []
  fy_bem_bdry = []
  with open(shape + '_bem' + eps_2 + '.txt', 'r') as f_bem:
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
        fx_bem_bdry.append(float(words[3]))
        fy_bem_bdry.append(float(words[4]))

  h_inv_fem = []
  fx_fem = []
  fy_fem = []
  fx_fem_bdry = []
  fy_fem_bdry = []
  with open(shape + '_fem' + eps_2 + '.txt', 'r') as f_fem:
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
        fx_fem_bdry.append(float(words[3]))
        fy_fem_bdry.append(float(words[4]))

  h_bem = 1. / np.array(h_inv_bem[:-1])
  exact = np.array([fx_bem[-1], fy_bem[-1]])
  data_bem = np.array([fx_bem[:-1], fy_bem[:-1]])
  data_bem_bdry = np.array([fx_bem_bdry[:-1], fy_bem_bdry[:-1]])
  error_bem = np.linalg.norm(data_bem - exact.reshape((2, 1)), axis=0)
  error_bem /= np.linalg.norm(exact)
  error_bem_bdry = np.linalg.norm(data_bem_bdry - exact.reshape((2, 1)), axis=0)
  error_bem_bdry /= np.linalg.norm(exact)
  fit_bem = np.polyfit(np.log(h_bem), np.log(error_bem), 1)
  print(f'Pullback approach (BEM): algebraic convergence with rate {fit_bem[0]:.3}')
  fit_bem_bdry = np.polyfit(np.log(h_bem), np.log(error_bem_bdry), 1)
  print(f'Stress tensor (BEM): algebraic convergence with rate {fit_bem_bdry[0]:.3}')

  last = 0
  for i in range(len(h_inv_bem)-1):
    if h_inv_fem[i] < h_inv_bem[-1]:
      last += 1
  h_fem = 1. / np.array(h_inv_fem[:last])
  data_fem = np.array([fx_fem[:last], fy_fem[:last]])
  data_fem_bdry = np.array([fx_fem_bdry[:last], fy_fem_bdry[:last]])
  error_fem = np.linalg.norm(data_fem - exact.reshape((2, 1)), axis=0)
  error_fem /= np.linalg.norm(exact)
  error_fem_bdry = np.linalg.norm(data_fem_bdry - exact.reshape((2, 1)), axis=0)
  error_fem_bdry /= np.linalg.norm(exact)
  fit_fem = np.polyfit(np.log(h_fem), np.log(error_fem), 1)
  print(f'Volume formula (FEM): algebraic convergence with rate {fit_fem[0]:.3}')
  fit_fem_bdry = np.polyfit(np.log(h_fem), np.log(error_fem_bdry), 1)
  print(f'Stress tensor (FEM): algebraic convergence with rate {fit_fem_bdry[0]:.3}')

  fig, ax = plt.subplots()
  ax.loglog(h_bem, h_bem**fit_bem[0] * np.exp(fit_bem[1]),
            color='silver', linestyle='--')
  ax.loglog(h_bem, h_bem**fit_bem_bdry[0] * np.exp(fit_bem_bdry[1]),
            color='silver', linestyle='--')
  ax.loglog(h_fem, h_fem**fit_fem[0] * np.exp(fit_fem[1]),
            color='silver', linestyle='--')
  ax.loglog(h_fem, h_fem**fit_fem_bdry[0] * np.exp(fit_fem_bdry[1]),
            color='silver', linestyle='--')
  ax.loglog(h_bem, error_bem, 'o-',
            label='Pullback approach (BEM)')
  ax.loglog(h_bem, error_bem_bdry, 'o-',
            label='Stress tensor (BEM)')
  ax.loglog(h_fem, error_fem, 'o-',
            label='Volume formula (FEM)')
  ax.loglog(h_fem, error_fem_bdry, 'o-',
            label='Stress tensor (FEM)')

  ax.set_xlabel('h')
  ax.set_ylabel('Relative error')
  #ax.set_title(shape.capitalize() + '-shaped ' + r'$\Omega_1$')
  ax.legend(loc='lower right')
  plt.savefig(shape + eps_2 + '_result.png', dpi=300)
  plt.show()