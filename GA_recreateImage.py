# # import math
# #
# # import cv2
# # import itertools
# # import functools
# # import numpy
# # import operator
# # import random
# #
# # ##############################################
# def my_psnr(I1, I2):
#     mse = numpy.mean(numpy.abs(I1 - I2))
#     if mse == 0:
#         return float('inf')
#     max_pixel = 255.0
#     psnr = (max_pixel / numpy.sqrt(mse))
#
#     return psnr
#
# #MSE fitness function but mutiply to -1 because higher is worser
# def fit_func(org_img, created_img):
#     similarity = my_psnr(org_img,created_img)
#     return similarity
# # ################################################
# #
# # #
# def crossover(parent1, parent2):
#     crossover_point = random.randint(math.floor(len(parent1)/3), math.floor(2*len(parent1)/3))
#     child1 = numpy.concatenate((parent1[:crossover_point] ,parent2[crossover_point:]))
#     child2 =numpy.concatenate((parent2[:crossover_point] , parent1[crossover_point:]))
#     return child1, child2
# #
# def mutate(chromosome, mutation_rate):
#     num_changes = int(chromosome.size * mutation_rate)
#     indices_to_change = numpy.random.choice(chromosome.size, num_changes, replace=False)
#     random_values = numpy.random.randint(0,255,size=num_changes)
#     mutated_chromosome = numpy.copy(chromosome)
#     numpy.put(mutated_chromosome, indices_to_change, random_values)
#     return mutated_chromosome
# #
# # def generate_next_generation(parents, mutation_rate,size_population,shape):
# #
# #     next_generation = []
# #     for member in parents:
# #         next_generation.append(member)
# #
# #     while len(next_generation) < size_population:
# #
# #         parent1 = random.choice(parents)
# #         parent2 = random.choice(parents)
# #         child1, child2 = crossover(parent1, parent2)
# #         child1 = mutate(child1, mutation_rate)
# #         child2 = mutate(child2, mutation_rate)
# #         next_generation.append(child1)
# #         next_generation.append(child2)
# #     next_generation=numpy.array(next_generation)
# #     return next_generation
# #     #return crossover(parents,shape)
# #
# #

# original_img = cv2.imread('Picture1.jpg')
#
# # change rgb image to vector(encoding stage)
# original_chromosome = original_img.flatten()
# #
# # # Population size
# # size_population = 12
# #
# # # Mating pool size
# # how_many_parents_use = 4
# #
# # # Mutation percentage
# # mutation_rate = 0.01
# #
# # ########################################################################
# # # Creating first population
# # next_pop = numpy.empty(shape=(size_population, numpy.size(original_img.flatten())),dtype=numpy.uint8)
# #
# # for i in range(size_population):
# #        next_pop[i, :] = numpy.random.random(numpy.size(original_img.flatten())) * 256
# #
# # #########################################################################
# #
# # #criterion for stop is number of iteration
# # for iteration in range(200000):
# #     print(iteration)
# #     # how much it is fitted#
# #     fitness_degrees = numpy.zeros(next_pop.shape[0])
# #     for i in range(next_pop.shape[0]):
# #         fitness_degrees[i] = fit_func(original_chromosome, next_pop[i, :])
# #     ##########
# #
# #     #Choosing best parents#
# #     parents = numpy.zeros((how_many_parents_use, next_pop.shape[1]), dtype=numpy.uint8)
# #     for i in range(how_many_parents_use):
# #          max_fitness_index = numpy.argmax(fitness_degrees)
# #          parents[i, :] = next_pop[max_fitness_index, :]
# #          fitness_degrees = numpy.delete(fitness_degrees, max_fitness_index)

# #     next_pop = generate_next_generation(parents, mutation_rate,size_population,original_img.shape)

# Display the final generation
# # si=int(numpy.size(next_pop)/8)
# # si_arr=numpy.uint8(next_pop[0,:si])
# #
# # #change vector to 3d numpy array
# # cv2.imshow("Image1",numpy.reshape(si_arr, original_img.shape))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

#############################################################قسمت بالا پیاده سازی بدون کمک کتابخانه pygad#####################################################
import cv2
import matplotlib
import numpy
import pygad
first_img=numpy.array(cv2.imread("Picture1.jpg"))
original_inputs = numpy.array(cv2.imread("Picture1.jpg"),dtype=float).flatten()
original_inputs=original_inputs/255#normalized

def fitness_func(t, solution, solution_idx):
    output = numpy.mean(numpy.abs(original_inputs-solution))
    fitness = (1 / numpy.sqrt(output))
    return fitness


fitness_function = fitness_func
num_generations = 1000
num_parents_mating = 4
sol_per_pop = 8
num_genes = original_inputs.size
init_range_low = 0.0
init_range_high = 1.0
parent_selection_type = "sss"
keep_parents = 1
crossover_type = "single_point"
mutation_type = "random"
random_mutation_min_val=0.0
random_mutation_max_val=1.0
mutation_percent_genes = 0.02

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       random_mutation_max_val=random_mutation_max_val,
                       random_mutation_min_val=random_mutation_min_val,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       mutation_by_replacement=True,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       )

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()

prediction = numpy.sum(numpy.array(original_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

# Scale the array between 0 and 255
min_val = numpy.min(solution)
max_val = numpy.max(solution)
scaled_array = (solution - min_val) * (255 / (max_val - min_val))
scaled_array=numpy.uint8(scaled_array)


#change vector to 3d numpy array
cv2.imshow("Image1",numpy.reshape(scaled_array, first_img.shape))
cv2.waitKey(0)
cv2.destroyAllWindows()