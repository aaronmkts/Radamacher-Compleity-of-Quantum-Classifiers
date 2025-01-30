import numpy as np
import matplotlib.pyplot as plt

#x = [2, 4, 6, 8, 10, 12, 14, 16]
#x = [2, 4, 8, 16]
x=np.linspace(10, 101, 10, dtype=int)
deez = len(x)
runs = 20

angl = np.load("angle_losses_num.npy")

a = angl

adv_train = np.zeros((runs, deez))
adv_test = np.zeros((runs, deez))
train = np.zeros((runs, deez))
test = np.zeros((runs, deez))

for i in range(runs):
    for j in range(deez):
        adv_train[i][j] = a[i][j][0]
        train[i][j] = a[i][j][1]
        adv_test[i][j] = a[i][j][2]
        test[i][j] = a[i][j][3]


#print(np.abs(adv_test-adv_train))
#print(adv_test)
#print(train)

adv_gen = np.mean(np.abs(adv_test-adv_train), 0)
adv_err = np.var(np.abs(adv_test-adv_train), 0)
gen = np.mean(np.abs(test-train), 0)
err = np.var(np.abs(test-train), 0)

diff = np.mean(adv_test-adv_train-test+train, 0)

#print(np.shape(adv_gen))

plt.errorbar(x, adv_gen, yerr=adv_err, label='Adv_Gen')
plt.errorbar(x, gen, yerr=err, label='Gen')
plt.title('Angle embedding')
#plt.plot(x, adv_gen-gen, label='Adv_Gen-Gen')
plt.legend()
plt.xlabel('data dimension')
plt.show()