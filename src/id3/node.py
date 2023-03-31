
class Node:
    """ A node in a tree. Each node has a name, a parent, and a list of children. """
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.data = None

    def __repr__(self):
        return self.name

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def remove_child(self, child):
        self.children.remove(child)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self, property_name):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        if self.data is not None:
            print(prefix + self.name + " (" + property_name + ": " + self.data + ")")
        else:
            print(prefix + self.name)
        if self.children:
            for child in self.children:
                child.print_tree(property_name)


if __name__ == "__main__":
    root = Node("Animal", None)
    legs = Node("Num Legs", root)
    legs.add_child(Node("<2", legs))
    legs.add_child(Node("=2", legs))
    legs.add_child(Node(">2", legs))
    color = Node("Color", root)
    color.add_child(Node("Red", color))
    color.add_child(Node("Blue", color))
    root.add_child(legs)
    root.add_child(color)


    root.print_tree("legs")