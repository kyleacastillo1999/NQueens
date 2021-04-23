from random import randint, shuffle
from point import Point
import math
import time

"""
Created with Pycharm
Author: Kyle Castillo
Date: 02/05/2021
Contact: kylea.castillo1999@gmail.com

Variation of the 8 Queens problem, N - Queens
Queens will be encoded as a Q while barriers are
encoded as a #. Empty spaces will be _.
"""


class Queens:

    # Constructor
    def __init__(self, height, width, queens, punishment, mutation_rate, execution_limit, elite_parent_per, size, goal,
                 max_iterations):
        self.height = height
        self.width = width
        self.queens = queens
        self.punishment = punishment
        self.mutation_rate = mutation_rate
        self.execution_limit = execution_limit
        self.elite_per = elite_parent_per
        self.pop_size = size
        self.goal = goal
        self.max_iterations = max_iterations
        self.current_gen = []
        self.next_gen = []
        self.gen_max_score = -99
        self.max_score = -99
        self.gen_avg = 0
        self.scores = []
        self.mutation_total_gen = 0
        self.mutation_total_all = 0
        self.__iterations = 0
        self.__best_member = []
        self.__best_member_gen = -1
        self.__best_member_id = -1
        self.__heuristic = 9
        self.__old_mutation_rate = self.mutation_rate
        self.__max_next_gen_pop = self.pop_size / 2
        self.__border_pt1 = Point(0, 0)
        self.__border_pt2 = Point(2, 4)
        self.__border_pt3 = Point(2, 3)
        self.__sky_net_cost = 9999999999999
        self.board = [["_"] * self.width for _ in range(self.height)]
        self.__generate_barriers()

    """
    __generate_barriers is a helper function used to generate the barriers for the board.
    Note that these barriers are hard encoded as a part of the problem.
    """

    def __generate_barriers(self):
        self.board[self.__border_pt1.get_y()][self.__border_pt1.get_x()] = "#"
        self.board[self.__border_pt2.get_y()][self.__border_pt2.get_x()] = "#"
        self.board[self.__border_pt3.get_y()][self.__border_pt3.get_x()] = "#"

    """
    reset_board is used to reset the board between iterations of testing the queens.
    """

    def __reset_board(self):
        self.board = [["_"] * self.width for _ in range(self.height)]
        self.__generate_barriers()

    def __set_punishment(self, new_score):
        self.punishment = new_score

    def get_gen_avg(self):
        return self.gen_avg

    """
    set_execution_limit sets a new execution limit for the threshold needed to breed.
    """

    def set_execution_limit(self, new_limit):
        self.execution_limit = new_limit

    def set_mutation_rate(self, new_rate):
        self.mutation_rate = new_rate

    def set_best(self, member, gen_id, member_id):
        """
        :param member: The member with the best queen placements.
        :param gen_id: The generation number of the best member
        :param member_id: The member number that performed the best.
        :return:
        """
        self.__best_member = member
        self.__best_member_gen = gen_id
        self.__best_member_id = member_id

    def get_iterations(self):
        return self.__iterations

    def set_iterations(self, itr):
        self.__iterations = itr

    """
    __destroy_pop is used to destroy the current population. Note this should only be used
    after the next generation is saved.
    """

    def __destroy_pop(self):
        self.current_gen = []

    """
    __destroy_next is used to destroy the next generation. Note this should only be used after
    the next generation has bred.
    """

    def __destroy_next_gen(self):
        self.next_gen = []

    """
    create_genesis_pop creates a list of randomly generated agents for the N-Queens
    problem. For the starting population no intelligence is needed other then the knowledge that
    the final value in the list represents the column with the barrier. For this another queen can be placed.
    """

    def create_random_pop(self):
        """
        :return: A 2D array where each value holds coordinates of where they are on the board.
        """

        # For the maximum population size create a member of the population.
        for i in range(self.pop_size - len(self.current_gen)):
            # Empty member list, the list represents where the member places their queens.
            member = []
            top = False

            # Coin toss: 0 is generate a number above. Otherwise generate it below.
            coin = randint(0, 1)

            # For each slot in the agent randomly assign a y coordinate, the x coordinate dictates what column it is in.
            for j in range(self.width + 1):
                # If we are dealing with the first column generate a random number between 1 and 7
                if j == 0:
                    rand_y = randint(1, self.height - 1)
                    queen_pt = Point(j, rand_y)
                    member.append(queen_pt)

                # If we are dealing with the row with barriers generate a number for the top or bottom part.
                elif j == 2:

                    # Top generation
                    if coin == 0:
                        rand_y = randint(0, 2)
                        queen_pt = Point(j, rand_y)
                        member.append(queen_pt)
                        top = True

                    # Bottom generation
                    else:
                        rand_y = randint(5, self.height - 1)
                        queen_pt = Point(j, rand_y)
                        member.append(queen_pt)

                # This is the special case for the third column, the last value in the list represents the bottom or top
                # Generation. This is based on the previous generation for the maze.
                elif j == 8:
                    if top:
                        # Generate on the bottom half
                        rand_y = randint(5, self.height - 1)
                        queen_pt = Point(2, rand_y)
                        member.append(queen_pt)
                    else:
                        # Generate on the top half
                        rand_y = randint(0, 2)
                        queen_pt = Point(2, rand_y)
                        member.append(queen_pt)

                # If we aren't dealing with a special case just generate a random position.
                else:
                    rand_y = randint(0, self.height - 1)
                    queen_pt = Point(j, rand_y)
                    member.append(queen_pt)

            # Append the new list as an agent of the population.
            self.current_gen.append(member)
        return self.current_gen

    """
    fitness checks each member of the population to test how well it preforms.
    The aim of the member is to get a heuristic of 9. The following equation
    will be used to evaluate the fitness score: numQueens - (punishment * intersections)
    """

    def fitness(self, member):

        # All members start with a score of 9 and are evaluated, less intersections mean a higher score.
        score = self.__heuristic
        intersections = 0

        # Go through each of the member's queen placements.
        for i in range(len(member)):
            test_queen = member[i]
            # Check the horizontal and diagonal parts to see if there are intersections with any other queens.
            # Note due to how the members only place one queen per column there will be no vertical intersections.
            for j in range(len(member)):
                compare_queen = member[j]
                # Check to make sure the point isn't referring to itself.
                if i != j:
                    # Check to see if the comparison is happening around a barrier
                    if test_queen.get_y() == self.__border_pt2.get_y() and \
                            compare_queen.get_y() == self.__border_pt2.get_y():
                        # Check to see if the points are on the same side.
                        if test_queen.get_x() <= 1 and compare_queen.get_x() <= 1:
                            if test_queen.get_y() == compare_queen.get_y():
                                # print('H-Intersection ', test_queen, ' and ', compare_queen)
                                intersections += 1
                        # Check to see if the points are on the same side.
                        if test_queen.get_x() >= 2 and compare_queen.get_x() >= 2:
                            if test_queen.get_y() == compare_queen.get_y():
                                # print('H-Intersection ', test_queen, ' and ', compare_queen)
                                intersections += 1

                    # Check to see if the comparison is happening around a barrier
                    elif test_queen.get_y() == self.__border_pt3.get_y() and \
                            compare_queen.get_y() == self.__border_pt3.get_y():
                        # Check to see if the points are on the same side.
                        if test_queen.get_x() <= 1 and compare_queen.get_x() <= 1:
                            if test_queen.get_y() == compare_queen.get_y():
                                # print('H-Intersection ', test_queen, ' and ', compare_queen)
                                intersections += 1
                        # Check to see if the points are on the same side.
                        if test_queen.get_x() >= 2 and compare_queen.get_x() >= 2:
                            if test_queen.get_y() == compare_queen.get_y():
                                # print('H-Intersection ', test_queen, ' and ', compare_queen)
                                intersections += 1

                    # If its not compare the y values and if they are the same.
                    else:
                        if test_queen.get_y() == compare_queen.get_y():
                            # print('H-Intersection ', test_queen, ' and ', compare_queen)
                            intersections += 1

                    # Check the diagonals, where the slope is equal to 1.
                    if abs(test_queen.slope(compare_queen)) == 1:
                        # Check the slope of the barrier
                        if not (abs(test_queen.slope(self.__border_pt2)) == 1
                                and abs(compare_queen.slope(self.__border_pt2)) == 1
                                or
                                abs(test_queen.slope(self.__border_pt3)) == 1
                                and abs(compare_queen.slope(self.__border_pt3) == 1)):
                            intersections += 1

                        # The slope is the same check to see if its on the same side, if it is then increment.
                        else:
                            if test_queen.get_x() >= 1 and compare_queen.get_x() >= 1:
                                intersections += 1
                            elif test_queen.get_x() < 2 and compare_queen.get_x() < 2:
                                intersections += 1

        # The intersections are divided by 2 due to the intersections being counted twice.
        return score - (intersections / 2 * self.punishment)

    """
    select_parents examines the score that was given to the parent and compares it to the threshold.
    If the parent is above or equal to the threshold it is added to the next generation to breed.
    """

    def select_parents(self):
        # First check to see if any parents could survive.
        if self.gen_max_score < self.execution_limit:
            # No parents would survive, reduce below the current population average.
            self.execution_limit = math.trunc(self.gen_avg)

            # Grab the best performing parents first.
            elite_count = self.elite_per
            while elite_count > 0:
                # Find the index of the best performing parents and add them to our list.
                elite_index = self.scores.index(max(self.scores))
                self.scores.pop(elite_index)
                self.next_gen.append(self.current_gen.pop(elite_index))
                elite_count -= 1

        # Go through the rest of the scores.
        for i in range(len(self.scores)):
            # Check to see which parents can breed.
            if self.scores[i] >= self.execution_limit:
                next_gen_parent = self.current_gen[i]
                self.next_gen.append(next_gen_parent)

        # Once you are done, clear the scores and update the execution limit.
        self.scores = []
        self.execution_limit += 0.5

        # Never go above the heuristic for the execution limit.
        if self.execution_limit == self.__heuristic:
            self.execution_limit = 8.5

    """
    crossover is called to select the parents of the last generation that were fit to breed, randomly shuffles the pop
    then crosses them twice, once with a 4/5 split and once with a 5/4 split. Creating two children.
    """

    def crossover(self):

        # Check to see if the next generation's length exceeds the limit of 500
        if len(self.next_gen) > self.__max_next_gen_pop:
            print('Survival candidates exceed environment threshold:', self.__max_next_gen_pop, ', culling.')
            # Check the difference then subtract.
            diff = len(self.next_gen) - self.__max_next_gen_pop
            while diff > 0:
                self.next_gen.pop()
                diff -= 1

        # Shuffle the next generation
        shuffle(self.next_gen)

        # Make two children, one using the first half of the genes from the first parent
        next_gen_len = len(self.next_gen)

        # Not the best way to got about crossbreeding but it works.
        for i in range(next_gen_len - 1):
            child_1 = []
            child_2 = []
            parent_1 = self.next_gen[i]
            parent_2 = self.next_gen[i + 1]
            # Child 1
            child_1.append(parent_1[0])
            child_1.append(parent_1[1])
            child_1.append(parent_1[2])
            child_1.append(parent_2[3])
            child_1.append(parent_2[4])
            child_1.append(parent_2[5])
            child_1.append(parent_2[6])
            child_1.append(parent_2[7])
            child_1.append(parent_1[8])  # Since the eighth value related to the third column it goes with parent 1

            # Check for a mutation
            mutation_check = randint(0, self.mutation_rate)
            if mutation_check == 0:
                self.mutation_total_all += 1
                self.mutation_total_gen += 1
                self.__mutate(child_1)

            # Child 2
            child_2.append(parent_2[0])
            child_2.append(parent_2[1])
            child_2.append(parent_2[2])
            child_2.append(parent_1[3])
            child_2.append(parent_1[4])
            child_2.append(parent_1[5])
            child_2.append(parent_1[6])
            child_2.append(parent_1[7])
            child_2.append(parent_2[8])  # Since the eighth value related to the third column it goes with parent 2

            # Check for a mutation
            mutation_check = randint(0, self.mutation_rate)
            if mutation_check == 0:
                self.mutation_total_all += 1
                self.mutation_total_gen += 1
                self.__mutate(child_2)

            self.current_gen.append(child_1)
            self.current_gen.append(child_2)

        self.create_random_pop()
        # Shuffle the current gen one more time.
        shuffle(self.current_gen)

        # Clear the next generation to have it empty for next generation.
        self.__destroy_next_gen()

    """
    __mutates is a helper function that alters one of the genes of the child.
    """

    def __mutate(self, child):
        mutated_index = randint(0, self.width)
        mutated_x = mutated_index

        # If its near the first barrier generate from 1-7 for X
        if mutated_index == 0:
            mutated_y = randint(1, self.height - 1)

        # If its near the barrier and the mutated_x is = 2 check to see where the y is for the child
        elif mutated_index == 2:
            mutated_x = 2
            if child[mutated_index].get_y() <= 2:
                mutated_y = randint(0, 2)
            else:
                mutated_y = randint(5, self.height - 1)

        # If its the last value check to see where it decided to generate
        elif mutated_index == 8:
            mutated_x = 2
            if child[mutated_index].get_y() <= 2:
                mutated_y = randint(0, 2)
            else:
                mutated_y = randint(5, self.height - 1)

        # No special cases, generate normally.
        else:
            mutated_y = randint(0, self.height - 1)

        # Create the gene and modify the child.
        mutated_pt = Point(mutated_x, mutated_y)
        child[mutated_index] = mutated_pt

    """
    __goal_check examines the generation average against the heuristic, if its within the goal then the algorithm
    terminates early.
    """

    def __goal_check(self):
        check = self.__heuristic - self.gen_avg
        if check < self.goal:
            print('Congrats! The algorithm is bred well enough to succeed!')
            print("Best preforming parent: ", self.__best_member_id, " at generation ", self.__best_member_gen)
            print("Max fitness: ", self.max_score)
            self.print_member_placement(self.__best_member)
            time.sleep(5)
            exit(1)

        elif self.max_iterations == self.__iterations:
            print("Best preforming parent: ", self.__best_member_id, " at generation ", self.__best_member_gen)
            print("Max fitness: ", self.max_score)
            self.print_member_placement(self.__best_member)
            time.sleep(5)
            exit(-1)

    """
    output_eval is used to help visualize how the queens are being placed onto the board
    and the results of the fitness function regarding the placement of the queens.
    """

    def output(self):
        self.gen_avg = 0
        self.gen_max_score = 0
        self.mutation_total_gen = 0
        # Loop through every parent in the current generation
        for i in range(len(self.current_gen)):
            print("Generation ", self.get_iterations(), " Member: ", i)
            # Check the individual's queen placement.
            for j in range(self.width + 1):
                test_pt = self.current_gen[i][j]
                pos = test_pt.get_y()
                if j != 8:
                    self.board[pos][j] = "Q"

                # For the final value of the member Q go back to column two to place the Queen.
                else:
                    self.board[pos][2] = "Q"

            # Output the information back to the console.
            self.print_board()

            # Evaluate the score and store it, the index of the score is the same as the queen.
            score = self.fitness(self.current_gen[i])
            self.scores.append(score)

            # Update generation average.
            self.gen_avg += score

            # Check to see if the score is greater then the all time max, it is save the member
            if score >= self.max_score:
                self.max_score = score
                self.set_best(self.current_gen[i], self.get_iterations(), i)

            # Check to see if the score is greater then the generation max.
            if score > self.gen_max_score:
                self.gen_max_score = score

            # Output fitness score and current generation max.
            print('Fitness Score: ', score)
            print('Generation Max: ', self.gen_max_score)

            # Check to see if the generation max exceeds the all time max.
            if self.gen_max_score > self.max_score:
                self.max_score = self.gen_max_score

            # Output the survival threshold.
            print('Current Survival Threshold: ', self.execution_limit)

            # Reset the board for the next individual.
            print('\n--------------------------------\n')
            self.__reset_board()

        # Calculate the generation average.
        self.gen_avg = self.gen_avg / self.pop_size

        # Output information back to the console.
        print('Generation Average :', self.gen_avg)
        print('Max fitness all time: ', self.max_score)
        print('Generation mutation: ', self.mutation_total_gen)
        print('Mutation Total all time: ', self.mutation_total_all)
        print('')

        # Check to see if the heuristic has been met.
        self.__goal_check()

        # Choose which parents will be available to breed.
        self.select_parents()

        # Output the number of individuals that can be parents.
        print(len(self.next_gen), " possible candidates for breeding")

        # Clear the current population and create children
        self.__destroy_pop()
        self.crossover()
        print('\n=================================\n')
        print('')

    # Prints an empty board with the barriers.
    def print_board(self):
        for row in range(self.width):
            for col in range(self.height):
                print(self.board[row][col], end='')
            print('')
        return None

    # Prints the placement of queens given a member.
    def print_member_placement(self, member):
        for pt in member:
            if pt.get_x() != 8:
                self.board[pt.get_y()][pt.get_x()] = "Q"
            else:
                self.board[pt.get_y][2] = "Q"
        for row in range(self.width):
            for col in range(self.height):
                print(self.board[row][col], end='')
            print('')
        self.__reset_board()
        return None


def main():
    # Adjust below to set the maximum population and mutation rate
    max_pop = 1000
    mutation_rate = 100  # Sets the odds, IE 1 / mutation_rate
    queens = 9
    punishment = 1
    iterations = 0

    execution_limit = -5  # set the minimum score needed to breed later on.
    max_iterations = 100

    elite_parents = 5  # Number of elite parents, parents that are guaranteed to breed.
    goal = 0.05  # Percentage of the population that needs to survive.

    test = Queens(8, 8, queens, punishment, mutation_rate, execution_limit, elite_parents, max_pop, goal,
                  max_iterations)
    test.create_random_pop()

    while test.get_iterations() <= max_iterations:
        test.output()
        test.set_iterations(iterations)
        iterations += 1


if __name__ == "__main__":
    main()
