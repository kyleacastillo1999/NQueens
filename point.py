"""
Created with Pycharm
Author: Kyle Castillo
Date: 02/05/2021
Contact: kylea.castillo1999@gmail.com
"""

"""
Point class to assist with returning the path information.
Stores the point itself as well as a reference to a parent node.
This helps build a tree of points to be used in the maze
to record the path that is taken.
"""


class Point:
    # Constructor
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.__parent = parent

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    # For tree creation
    def get_parent(self):
        return self.__parent

    def set_parent(self, parent):
        self.__parent = parent

    def slope(self, point):
        if point.get_x() - self.x == 0:
            return 0
        else:
            return (point.get_y() - self.y) / (point.get_x() - self.x)

    def __str__(self):
        return "Point(%s,%s)" % (self.x, self.y)
