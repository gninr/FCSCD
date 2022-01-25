from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

if __name__ == '__main__':
  u_x_bem = []
  u_y_bem = []
  u_bem = []
  psi_x_bem = []
  psi_y_bem = []
  psi_bem = []
  with open('transmission_bem.txt', 'r') as f_bem:
    Dir = False
    Neu = False
    for line in f_bem:
      words = line.split()
      if words[0] == 'x':
        continue
      if not Dir:
        if words[0] == 'Dirichlet':
          Dir = True
      else:
        if not Neu:
          if words[0] == 'Neumann':
            Neu = True
          else:
            u_x_bem.append(float(words[0]))
            u_y_bem.append(float(words[1]))
            u_bem.append(float(words[2]))
        else:
          psi_x_bem.append(float(words[0]))
          psi_y_bem.append(float(words[1]))
          psi_bem.append(float(words[2]))

  u_x_fem = []
  u_y_fem = []
  u_fem = []
  psi_1_x_fem = []
  psi_1_y_fem = []
  psi_1_fem = []
  psi_2_x_fem = []
  psi_2_y_fem = []
  psi_2_fem = []
  with open('transmission_fem.txt', 'r') as f_fem:
    Dir = False
    Neu = False
    for line in f_fem:
      words = line.split()
      if words[0] == 'x':
        continue
      if not Dir:
        if words[0] == 'Dirichlet':
          Dir = True
      else:
        if not Neu:
          if words[0] == 'Neumann':
            Neu = True
          else:
            u_x_fem.append(float(words[0]))
            u_y_fem.append(float(words[1]))
            u_fem.append(float(words[2]))
        else:
          if (words[3] == '1'):
            psi_1_x_fem.append(float(words[0]))
            psi_1_y_fem.append(float(words[1]))
            psi_1_fem.append(float(words[2]))
          else:
            psi_2_x_fem.append(float(words[0]))
            psi_2_y_fem.append(float(words[1]))
            psi_2_fem.append(float(words[2]))

  
  ax1 = plt.subplot(121, projection='3d')
  ax1.scatter(u_x_bem, u_y_bem, u_bem, s=5, label='BEM')
  ax1.scatter(u_x_fem, u_y_fem, u_fem, s=5, label='FEM')
  ax1.legend()
  ax1.set_title('Dirichlet trace')

  ax2 = plt.subplot(122, projection='3d')
  ax2.scatter(psi_x_bem, psi_y_bem, psi_bem, s=5, label='BEM')
  ax2.scatter(psi_1_x_fem, psi_1_y_fem, psi_1_fem, s=5, label='FEM, inner')
  ax2.scatter(psi_2_x_fem, psi_2_y_fem, psi_2_fem, s=5, label='FEM, outer')
  ax2.legend()
  ax2.set_title('Neumann trace')
  

  '''
  ax = plt.subplot(projection='3d')
  ax.scatter(u_x_bem, u_y_bem, u_bem, label='BEM')
  ax.scatter(u_x_fem, u_y_fem, u_fem, label='FEM')
  ax.legend()
  ax.set_title('Dirichlet trace')
  '''
  '''
  ax = plt.subplot(projection='3d')
  ax.scatter(psi_x_bem, psi_y_bem, psi_bem, label='BEM')
  ax.scatter(psi_1_x_fem, psi_1_y_fem, psi_1_fem, label='FEM, inner')
  ax.scatter(psi_2_x_fem, psi_2_y_fem, psi_2_fem, label='FEM, outer')
  ax.legend()
  ax.set_title('Neumann trace')
  '''

  plt.show()
