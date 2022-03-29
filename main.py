import numpy as np
import random
import matplotlib.pyplot as plt
import time
from numba import njit
from datetime import datetime

###### в lost_request нужно смотреть на условие с ==, тк к без изменений оно жестко ограничивает количество загруженных устройств
# на 1 меньше общего количества
# FIXME сделать вызов различных распределений в зависимости от класса
print('enter number of channels')
C_initial = int(input())
C_initial += 1
print('enter number of classes >= 3')
K_initial = int(input())
t_global = 0.0
print('enter internal modelling time (time should be >= 300)')
t_ending = float(input())
print('enter delta_t')
delta_t = float(input())
tick_counter = 0
steps = int(t_ending / delta_t)
point_number = int(1 / delta_t)
# данные
stamps = []
C_current = []
# константы

N_k = np.zeros(K_initial)
for i in range(len(N_k)):
    print('enter source number N')
    N_k[i] = int(input())
N_k = np.array(N_k, dtype='int64')
print('N_k are', N_k)

nu_k = np.zeros(K_initial)
for i in range(len(nu_k)):
    print('enter source intensity nu')
    nu_k[i] = float(input())
nu_k = np.array(nu_k)
print('nu_k are ', nu_k)

b_k = np.ones(K_initial)
for i in range(len(b_k)):
    print('enter number of channels occupied by class b_k')
    b_k[i] = int(input())
b_k = np.array(b_k, dtype='int64')
print('b_k are ', b_k)

t_k = np.ones(K_initial)
for i in range(len(t_k)):
    print('enter t_k')
    t_k[i] = float(input())
t_k = np.array(t_k, dtype='float64')
print('t_k are ', t_k)

mu_k = np.ones(K_initial)
for i in range(len(mu_k)):
    print('enter distribution parameter mu_k')
    mu_k[i] = int(input())
mu_k = np.array(mu_k, dtype='float64')
print('mu_k are ', mu_k)
print('executing program...')


# массив с текущими запросами, распределение ядер по idшкам, id=0 - незанятый, время до выполнения запроса
# [id, N, exec_time, age]
request_handler = np.zeros((C_initial, 4), dtype='int64')

# количество различных запущенныъ процессов на классе
amount_of_processes = np.zeros(K_initial, dtype='int64')
# количество выполненных ДО КОНЦА запросов, нужно домножать на b_k для получения абсолютного значений
request_count = np.ones(K_initial, dtype='int64')
# количество потерянных
lost_request_count = np.ones(K_initial, dtype='int64')

source_module = []


# экспоненциальное распределение ТОЛЬКО ДЛЯ ВРЕМЕНИ, тк с округлением
@njit
def dist_wrap(lam, r):
    return (-np.log(r) / lam) * point_number


def t_dist(lam):
    r = random.random()
    return np.ceil(dist_wrap(lam, r)).astype('int64')


# время до первых запросов по классам
def t_N(nu):
    return t_dist(nu)


# 1 distribution
def t_execution_1(mu):
    return t_dist(mu)


def t_execution_const():
    return 1 * point_number


def t_execution_2(mu):
    r = random.random()
    if r > 0.5:
        return t_dist(0.5 * mu)
    else:
        #         print('rolled zero t = ', t_global)
        return 1


def t_execution_3(mu):
    r = random.random()
    if r < 0.33:
        return t_dist(0.66666)
    else:
        return t_dist(1.33333)


def t_execution_erlang(mu):
    return t_dist(mu) + t_dist(mu)


def lost_request(i):
    loss_state = False

    if C_initial - np.count_nonzero(request_handler[:, 0]) - int(t_k[i]) < b_k[i]:
        loss_state = True

    elif C_initial - np.count_nonzero(request_handler[:, 0]) - int(t_k[i]) == b_k[i]:
        if t_k[i] - float(int(t_k[i])) < np.random.rand(1):
            loss_state = True

    return loss_state


# запрос возврщается в источник
def losing_procedure_1(i, j):
    source_module[i][j, 0] = 1
    source_module[i][j, 1] = t_dist(nu_k[i])


def losing_procedure_2(i, j):
    max_age = 0
    N_temp = 0
    for k in range(C_initial):
        if request_handler[k, 0] == i + 1 and request_handler[k, 3] > max_age:
            max_age = request_handler[k, 3]
    cond_13 = (request_handler[:, 0] == i + 1) * (request_handler[:, 3] == max_age)
    for k in range(C_initial):
        if cond_13[k]:
            N_temp = request_handler[k, 1]
            request_handler[k, 0] = 0
            request_handler[k, 1] = 0
            request_handler[k, 2] = 0
            request_handler[k, 3] = 0
    source_module[i][N_temp, 0] = 1
    source_module[i][N_temp, 1] = t_dist(nu_k[i])
    b = b_k[i]
    exec_time = t_execution_1(mu_k[i])
    cond_9 = (request_handler[:, 0] == 0)
    for k in range(C_initial):
        if cond_9[k]:
            request_handler[k, 0] = i + 1
            request_handler[k, 1] = j
            request_handler[k, 2] = exec_time
            request_handler[k, 3] = 0
            b += -1
            if b == 0:
                break
    if b != 0:
        print("ERROR not enough space at id = ", i + 1)


@njit
def f_any_less_than_zero(arr, ind):
    cond = False
    for i in arr:
        if i[ind] <= 0:
            cond = True
            break
    return cond


@njit
def any_done(arr, ind_1, ind_2):
    cond = False
    for i in arr:
        if i[ind_1] != 0 and i[ind_2] <= 0:
            cond = True
            break
    return cond


# забиваем изначальные задержки в стек
for i in range(K_initial):
    source_module.append(np.zeros((N_k[i], 2), dtype='int64'))

# source_module[i][j, 0] - статус (1 - запрос ждет выдачи, 0 запрос обрабытывается)
# source_module[i][j, 1] - время до выдачи запроса
for i in range(len(source_module)):
    for j in range(len(source_module[i])):
        source_module[i][j, 0] = 1
        source_module[i][j, 1] = t_dist(nu_k[i])

start_time = time.time()
# ТЕЛО ПРОГРАММЫ
while t_global < t_ending:
    # перемещаем запросы с источника в обработчик
    for i in range(K_initial):
        if len(source_module[i]) != 0:
            if f_any_less_than_zero(source_module[i], 1):
                cond_7 = (source_module[i][:, 0] == 1) * (source_module[i][:, 1] <= 0)
                for j in range(N_k[i]):
                    if cond_7[j]:
                        source_module[i][j, 0] = 0

                        if lost_request(i):
                            losing_procedure_1(i, j)
                            lost_request_count[i] += 1


                        else:
                            b = b_k[i]
                            exec_time = t_execution_1(mu_k[i])
                            cond_9 = request_handler[:, 0] == 0
                            for k in range(C_initial):
                                if cond_9[k]:
                                    request_handler[k, 0] = i + 1
                                    request_handler[k, 1] = j
                                    request_handler[k, 2] = exec_time
                                    request_handler[k, 3] = 0
                                    b += -1
                                    if b == 0:
                                        break
                            if b != 0:
                                print("ERROR not enough space at id = ", i + 1)

    # убраем выполненные запросы обратно в источник
    if any_done(request_handler, 0, 2):
        cond_10 = (request_handler[:, 0] != 0) * (request_handler[:, 2] <= 0)
        for i in range(C_initial):
            if cond_10[i]:
                if source_module[request_handler[i, 0] - 1][request_handler[i, 1], 0] == 0:
                    temp_ind = request_handler[i, 0] - 1
                    temp_cell_num = request_handler[i, 1]
                    request_count[temp_ind] += 1
                    source_module[temp_ind][temp_cell_num, 0] = 1
                    source_module[temp_ind][temp_cell_num, 1] = t_dist(nu_k[temp_ind])
                request_handler[i, 0] = 0
                request_handler[i, 1] = 0
                request_handler[i, 2] = 0
                request_handler[i, 3] = 0

    if tick_counter % (steps // 400) == 0 and t_global > 250:
        #         stamps.append([t_global, np.count_nonzero(request_handler[:, 0] == 1), np.count_nonzero(request_handler[:, 0] == 2),
        #                       np.count_nonzero(request_handler[:, 0] == 3), C_current])
        stamps.append([t_global, float(lost_request_count[0]) / float(request_count[0] + lost_request_count[0]),
                       float(lost_request_count[1]) / float(request_count[1] + lost_request_count[1]),
                       float(lost_request_count[2]) / float(request_count[2] + lost_request_count[2]),
                       np.count_nonzero(request_handler[:, 0] != 0)])
        C_current.append([np.count_nonzero(request_handler[:, 0] == 1), np.count_nonzero(request_handler[:, 0] == 2),
                          np.count_nonzero(request_handler[:, 0] == 3)])

    # t ++
    for i in range(K_initial):
        if len(source_module[i]) != 0:
            source_module[i][:, 1] = source_module[i][:, 1] - 1
    request_handler[:, 2] = request_handler[:, 2] - 1
    request_handler[:, 3] = request_handler[:, 3] + 1
    t_global += delta_t
    tick_counter += 1

# количество ячеек,занятых обработкой запросов k-го типа
# print('request handler: ', request_handler)
# print('')
# for i in range(len(source_module)):
#     print(source_module[i])
print('request count: ', request_count, ' lost request count: ', lost_request_count)
print('')
print("--- %s seconds ---" % (time.time() - start_time))
print('mean C: ', np.mean(np.array(C_current), 0))
print(float(lost_request_count[0]) / float(request_count[0] + lost_request_count[0]),
      float(lost_request_count[1]) / float(request_count[1] + lost_request_count[1]),
      float(lost_request_count[2]) / float(request_count[2] + lost_request_count[2]))
stamps = np.array(stamps)
plt.figure(figsize=(15, 8))
plt.plot(stamps[:, 0], stamps[:, 1], label='id 1')
plt.plot(stamps[:, 0], stamps[:, 2], label='id 2')
plt.plot(stamps[:, 0], stamps[:, 3], label='id 3')
# print(np.array(C_current))
plt.xlabel('Time')
plt.ylabel('Amount of requests processing, timewise')
plt.legend()
date_time = datetime.now()
name_string = date_time.strftime("%m%d%Y_%H%M%S")
name_string = name_string + ".png"
plt.savefig(name_string)
