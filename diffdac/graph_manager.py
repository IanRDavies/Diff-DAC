import numpy as np


class GraphManager(object):
    """ Parent class from which graphs for tracking agents should inherit """
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.phone_book = self._make_graph()

    def _make_graph(self):
        """
        Returns a nested list of peers; the outer-list is indexed by rank,
        the inner list denotes the set of peers that 'rank' can send
        messages to at any point in time
        """
        raise NotImplementedError

    def get_peers(self):
        """ Returns the out and in-peers corresponding to 'self.rank' """
        raise NotImplementedError


class SymmetricConnectionGraph(GraphManager):
    """
    A graph that can have any topology expressed in a symmetric adjacency matrix.
    Adjacency matrix must be symmetric as we assume undirected links.
    """
    def __init__(self, adjacency_matrix, rank):
        # Validate the adjacency matrix (assumed to be a numpy array)
        assert adjacency_matrix.ndim == 2
        assert np.all(np.logical_or(adjacency_matrix == 0, adjacency_matrix == 1))
        assert np.all(adjacency_matrix == adjacency_matrix.T)
        assert np.all(np.diag(adjacency_matrix) == 0)
        world_size = adjacency_matrix.shape[0]
        assert rank < world_size
        self._adjacency_matrix = adjacency_matrix
        super().__init__(rank, world_size)

    def _make_graph(self):
        """
        Returns a nested list of peers; the outer-list is indexed by rank,
        the inner list denotes the set of peers that 'rank' can send
        messages to at any point in time
        """
        # Get indices where adjacency matrix is non-zero.
        phone_book = [list(np.where(a)[0]) for a in self._adjacency_matrix]
        return phone_book

    def get_peers(self):
        # This is easy to implement since the adjacency matrix is symmetric.
        return self.phone_book[self.rank], self.phone_book[self.rank]
