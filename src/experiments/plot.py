import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pennylane as qml
import torch
import math

from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "text.usetex": True,              # Use LaTeX to write all text
    "font.family": "serif",           # Use a serif font family
    "font.serif": ["Computer Modern Roman"],  # The default LaTeX font
})

d = [2, 4, 6, 8, 10]

#d = [2, 4, 8]
deez = len(d)
x=np.array(10+np.linspace(0, 45, 10), dtype=int)
teez = len(x)

step = (0.01-0.001)/7
"""
epsilons = np.linspace(0.001, 0.01, 8)
eez = len(epsilons)

noiseless = np.load("noiseless_losses_epsilon_d=6_const_classifier2.npy")
noisy = np.load("noisy_losses_epsilon_d=6_const_classifier2.npy")

#noiseless_th = np.load("noiseless_th.npy")
#noisy_th = np.load("noisy_th.npy")
a = noiseless


noiseless_adv = np.load("noiseless_adv_th.npy")
noiseless = np.load("noiseless_th.npy")
noisy_adv = np.load("noisy_adv_th.npy")
noisy_adv_loose = np.load("noisy_adv_th3.npy")
noisy = np.load("noisy_th.npy")


plt.figure(figsize=(4, 3))
plt.plot(epsilons, (noiseless_adv), color='red', label='Noiseless ARC bound (Theorem 3)')
plt.plot(epsilons, (noiseless), color='green', label='Noiseless RC bound (Theorem 2)')
plt.plot(epsilons, (noisy_adv), '--', color='red', label='Noisy ARC bound (Theorem 4)')
plt.plot(epsilons, (noisy), '--',color='green', label='Noisy RC bound (Theorem 2)')
#plt.plot(epsilons, noisy_adv_loose,'--', color='red')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel(r'$\epsilon$', fontsize=13)
plt.title('Generalization Error Bounds', fontsize=10)
plt.legend(fontsize=8, loc=2)
plt.subplots_adjust(bottom=0.15)
plt.savefig("eps_bounds", dpi=500)
plt.show()
"""

runs = 20


ampl = np.load("ampl_losses_dim_r=30_m=20_e=40.npy")
angl = np.load("angle_losses_dim_r=30_m=20_e=40.npy")

a = angl
"""
adv_train = np.zeros((runs, teez, deez))
adv_test = np.zeros((runs, teez, deez))
train = np.zeros((runs, teez, deez))
test = np.zeros((runs, teez, deez))

for i in range(runs):
    for j in range(teez):
        for k in range(deez):
            adv_train[i][j][k] = a[i][j][k][0]
            train[i][j][k] = a[i][j][k][1]
            adv_test[i][j][k] = a[i][j][k][2]
            test[i][j][k] = a[i][j][k][3]

adv_gen = np.mean(adv_test-adv_train, 0).swapaxes(0, 1)
adv_err = np.var(adv_test-adv_train, 0).swapaxes(0, 1)
gen = np.mean(test-train, 0).swapaxes(0, 1)
err = np.var(test-train, 0).swapaxes(0, 1)

plt.errorbar(x, np.mean(test, 0).swapaxes(0, 1)[0], yerr=np.var(test, 0).swapaxes(0, 1)[0], color='blue')
plt.errorbar(x, np.mean(train, 0).swapaxes(0, 1)[0], yerr=np.var(train, 0).swapaxes(0, 1)[0], color='green')
plt.xlabel("Number of Training Samples")
plt.show()

"""
"""
adv_test = np.load("adv_test_loss.npy")
test = np.load("test_loss.npy")

adv_train = np.load("adv_train_loss.npy")
train = np.load("train_loss.npy")

adv_gen = adv_test-adv_train
gen = test-train
print(adv_gen-gen)

plt.figure(figsize=(4, 3))
plt.plot(np.mean(adv_test, axis=0), color="red", label="Adversarial test loss")
plt.plot(np.mean(adv_train, axis=0), '--', color="red", label="Adversarial train loss")
plt.plot(np.mean(test, axis=0), color="green", label="Conventional test loss")
plt.plot(np.mean(train, axis=0), '--',color="green", label="Conventional train loss")
plt.xticks(np.linspace(0, 20, 11, dtype=int), fontsize=10)
plt.xlabel('Training epoch', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc=1, fontsize=8)
plt.subplots_adjust(bottom=0.15)
plt.savefig("motiv", dpi=500)
plt.show()
"""

adv_train = np.zeros((runs, deez))
adv_test = np.zeros((runs, deez))
train = np.zeros((runs, deez))
test = np.zeros((runs, deez))
rad = np.zeros((runs, deez))
adv = np.zeros((runs, deez))


for i in range(runs):
    for j in range(deez):
        adv_train[i][j] = a[i][j][0]
        train[i][j] = a[i][j][1]
        adv_test[i][j] = a[i][j][2]
        test[i][j] = a[i][j][3]
       
        x = np.load(f"x_r=30_d={d[j]}_m=20.npy")[i]
        device = qml.device("default.mixed", wires=d[j])
        d_H = int(np.ceil(np.log2(d[j])))
        
        @qml.qnode(device, interface="torch", diff_method="backprop")
        def embedding(inputs):
            #for k in range(d[j]):  # Apply angle embedding
             #   qml.RY(inputs[k], wires=k)
            qml.AmplitudeEmbedding(features=inputs, wires=range(d_H), normalize=True, pad_with=0.)
            return qml.density_matrix(range(d[j]))

        states = torch.tensor(np.array([np.power(embedding(inputs), 2.) for inputs in x]))
        rad[i][j] = 2.5*torch.sum(torch.sqrt(torch.abs(torch.linalg.eigvals(torch.sum(states, 0))))).numpy()/20

        adv[i][j] = 2 ** (1 + np.ceil(np.log2(d[j]))) * np.min([1, 0.3 * np.sqrt(d[j]) / np.min(np.sqrt(np.sum(np.power(x, 2), axis=1)))])


J = 36*(np.sqrt(np.log(6))/6+np.pi/2*(1-math.erf(np.sqrt(np.log(6)))))
excess = 2.5*J*np.mean(adv, axis=0)/np.sqrt(20)
#excess = 2.5*np.array([2*(2*0.3)**d[i]*J/np.sqrt(20) for i in range(deez)])


adv_gen = np.mean(adv_test-adv_train, 0)
adv_err = np.var(np.abs(adv_test-adv_train), 0)
gen = np.mean(test-train, 0)
err = np.var(np.abs(test-train), 0)

#print(np.size(adv_gen))

#diff = np.mean(adv_test-adv_train-test+train, axis=0)
diff = adv_gen - gen
diff_err = np.var(adv_test-adv_train-test+train, axis=0)

#adv_grad = np.log10(np.gradient(adv_gen-gen, step))
plt.figure(figsize=(4, 3))
#plt.errorbar(d, (adv_gen), color='red', label='Adversarial gen')
plt.errorbar(d, 2*(np.mean(rad, axis=0)+excess), color='red', label='Adversarial generalization error bound')
#plt.errorbar(d, gen, label='Conventional generalization error bound', color='green')
#plt.plot(d, adv_gen, label='Adversarial Generalization Error', color='red')

"""
a = ampl
for i in range(runs):
    for j in range(deez):
        adv_train[i][j] = a[i][j][0]
        train[i][j] = a[i][j][1]
        adv_test[i][j] = a[i][j][2]
        test[i][j] = a[i][j][3]

adv_gen = np.mean(adv_test-adv_train, 0)
adv_err = np.var(adv_test-adv_train, 0)
gen = np.mean(test-train, 0)
err = np.var(test-train, 0)
"""
#diff = np.mean(adv_test-adv_train-test+train, axis=0)
#diff = adv_gen - gen
#diff_err = np.var(adv_test-adv_train-test+train, axis=0)

#plt.errorbar(epsilons, diff, label='Noisy', color='orange')
#plt.plot(d, np.mean(test-train, 0), color='green')
plt.errorbar(d, 2*(np.mean(rad, axis=0)), color='green', label='Conventional generalization error bound')
#adv_grad = np.log10(np.gradient(adv_gen, step))
#plt.errorbar(epsilons, (adv_gen), linestyle='--', color='red', label='Noisy Adversarial')
#plt.errorbar(epsilons, (gen), linestyle='--', color='green', label='Noisy Conventional')

#noiseless_grad = np.log10(np.gradient(noiseless_th, step))
#noisy_grad = np.log10(np.gradient(noisy_th, step))

#plt.errorbar(epsilons, (noiseless_th), color='red', label='Gen')
#plt.errorbar(epsilons, (noisy_th), linestyle='--', color='orange', label='Gen')
#plt.legend()
#plt.errorbar(d, adv_gen, label='Adversarial generalization error', color='red')
#plt.errorbar(d, gen, label='Conventional generalization error', color='green')

#plt.plot(d, np.mean(adv_test-adv_train, axis=0), label='Angle')
#plt.plot(d, th, label = 'Theory')
plt.title('Amplitude Embedding', fontsize=10)
plt.xlabel(r'$d$', fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=8, loc=2)
plt.subplots_adjust(bottom=0.15)
#plt.ylabel('Accuracy', fontsize=13)
plt.xticks(np.linspace(2, 10, 5, dtype=int))

#plt.ylim((0.0925, 0.096))
plt.savefig("ampl_th", dpi=500)
plt.show()
