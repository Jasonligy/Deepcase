import logging
import numpy as np
import pickle
import scipy.sparse as sp
import torch
import warnings
from collections       import Counter
from sklearn.neighbors import KDTree
from tqdm              import tqdm

from .cluster import Cluster
from .utils   import group_by, unique_2d, sp_unique

# Set logger
logger = logging.getLogger(__name__)

class Interpreter(object):

    def __init__(self, context_builder, features, eps=0.1, min_samples=5,
                 threshold=0.2):
        """Interpreter for a given ContextBuilder.

            Parameters
            ----------
            context_builder : ContextBuilder
                ContextBuilder to interpret.

            features : int
                Number of different possible security events.

            eps : float, default=0.1
                Epsilon used for determining maximum distance between clusters.

            min_samples : int, default=5
                Minimum number of required samples per cluster.

            threshold : float, default=0.2
                Minimum required confidence of ContextBuilder before using a
                context in training clusters.
            """
        # Initialise ContextBuilder
        self.context_builder = context_builder

        # Create cluster algorithm dbscan
        self.dbscan = Cluster(p=1)

        # Set parameters
        self.features    = features
        self.eps         = eps
        self.min_samples = min_samples
        self.threshold   = threshold

        # Store entries
        self.clusters = np.zeros(0)
        self.vectors  = np.zeros((0, self.features))
        self.events   = np.zeros(0)
        self.tree     = dict()
        self.labels   = dict()

    ########################################################################
    #                         Fit/Predict methods                          #
    ########################################################################

    def fit(self,
            X,
            y,
            scores,
            iterations = 100,
            batch_size = 1024,
            strategy   = "max",
            NO_SCORE   = -1,
            verbose    = False,
        ):
        """Fit the Interpreter by performing clustering and assigning scores.

            Fit function is a wrapper that calls the following methods:
              1. Interpreter.cluster
              2. Interpreter.score_clusters
              3. Interpreter.score

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context to cluster.

            y : torch.Tensor of shape=(n_samples, 1)
                Events to cluster.

            scores : array-like of float, shape=(n_samples,)
                Scores for each sample in cluster.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            strategy : string (max|min|avg), default=max
                Strategy to use for computing scores per cluster based on scores
                of individual events. Currently available options are:
                - max: Use maximum score of any individual event in a cluster.
                - min: Use minimum score of any individual event in a cluster.
                - avg: Use average score of any individual event in a cluster.

            NO_SCORE : float, default=-1
                Score to indicate that no score was given to a sample and that
                the value should be ignored for computing the cluster score.
                The NO_SCORE value will also be given to samples that do not
                belong to a cluster.

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            self : self
                Returns self
            """
        # Call cluster method
        clusters = self.cluster(
            X          = X,
            y          = y,
            iterations = iterations,
            batch_size = batch_size,
            verbose    = verbose,
        )

        # Call score_clusters method to distribute individual scores over
        # clusters according to chosen strategy
        scores = self.score_clusters(
            scores   = scores,
            strategy = strategy,
            NO_SCORE = NO_SCORE,
        )

        # Set scores
        self.score(
            scores  = scores,
            verbose = verbose,
        )

        # Return self
        return self


    def predict(self, X, y, iterations=100, batch_size=1024, verbose=False):
        """Predict maliciousness of context samples.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context for which to predict maliciousness.

            y : torch.Tensor of shape=(n_samples, 1)
                Events for which to predict maliciousness.

            iterations : int, default=100
                Iterations used for optimization.

            batch_size : int, default=1024
                Batch size used for optimization.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Predicted maliciousness score.
                Positive scores are maliciousness scores.
                A score of 0 means we found a match that was not malicious.
                Special cases:

                * -1: Not confident enough for prediction
                * -2: Label not in training
                * -3: Closest cluster > epsilon
            """
        # Get unique samples
        X, y, inverse_result = unique_2d(X, y)

        ####################################################################
        #                         Compute vectors                          #
        ####################################################################

        # Compute vectors
        vectors, mask = self.attended_context(
            X           = X,
            y           = y,
            threshold   = self.threshold,
            iterations  = iterations,
            batch_size  = batch_size,
            verbose     = verbose,
        )

        # Initialise result
        result = np.full(vectors.shape[0], -4, dtype=float)

        ####################################################################
        #                   Find closest known sequences                   #
        ####################################################################

        # Group sequences by individual events
        events = group_by(y[mask].squeeze(1).cpu().numpy())
        print(type(events))
        # Add verbosity, if necessary
        if verbose: events = tqdm(events, desc="Predicting      ")
        count=0
        # Loop over all events
        for event, indices in events:

            ############################################################
            #                   Case - unknown event                   #
            ############################################################

            # If event is not in training set, set to -2
            if event not in self.tree:
                result[indices] = -2
                continue

            ############################################################
            #                    Case - known event                    #
            ############################################################
            if count==0:
                print(vectors.shape)
            # Get vectors for given event
            vectors_ = vectors[indices]
            
            # Get unique vectors - optimizes computation time
            vectors_, inverse, _ = sp_unique(vectors_)

            # Get closest cluster
            distance, neighbours = self.tree[event].query(
                X               = vectors_.toarray(),
                return_distance = True,
                dualtree        = vectors_.shape[0] >= 1e3, # Optimization
            )
            if count==0:
                print('distance')
                print(type(distance))
                print(distance.shape)
                print('neighbor')
                print(type(neighbours))
                print(neighbours.shape)
            # Get neighbour indices
            neighbours = self.tree[event].get_arrays()[1][neighbours][:, 0]
            if count==0:
                print('proces')
                print(neighbours.shape)
                print(neighbours)
            # Compute neighbour scores
            scores = np.asarray([
                self.labels[event][neighbour] for neighbour in neighbours
            ])

            ############################################################
            #               Set result, based on epsilon               #
            ############################################################

            # Set resulting indices
            result[indices] = np.where(
                distance[:, 0] <= self.eps, # Check if closest cluster > eps
                scores,                     # If so, assign actual score
                -3,                         # Else, closest cluster > eps, -3
            )[inverse]
         
            count+=1

        ####################################################################
        #                     Add non-confident events                     #
        ####################################################################

        result_ = np.full(X.shape[0], -1, dtype=float)
        result_[mask.cpu().numpy()] = result
        result = result_
        print('result')
        print(result.shape)
        print(result[inverse_result.cpu().numpy()].shape)
        # Return result
        return result[inverse_result.cpu().numpy()]


    def fit_predict(self,
            X,
            y,
            scores,
            iterations = 100,
            batch_size = 1024,
            strategy   = "max",
            NO_SCORE   = -1,
            verbose    = False,
        ):
        """Fit Interpreter with samples and labels and return the predictions of
            the same samples after running them through the Interpreter.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context to cluster.

            y : torch.Tensor of shape=(n_samples, 1)
                Events to cluster.

            scores : array-like of float, shape=(n_samples,)
                Scores for each sample in cluster.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            strategy : string (max|min|avg), default=max
                Strategy to use for computing scores per cluster based on scores
                of individual events. Currently available options are:
                - max: Use maximum score of any individual event in a cluster.
                - min: Use minimum score of any individual event in a cluster.
                - avg: Use average score of any individual event in a cluster.

            NO_SCORE : float, default=-1
                Score to indicate that no score was given to a sample and that
                the value should be ignored for computing the cluster score.
                The NO_SCORE value will also be given to samples that do not
                belong to a cluster.

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            result : np.array of shape=(n_samples,)
                Predicted maliciousness score.
                Positive scores are maliciousness scores.
                A score of 0 means we found a match that was not malicious.
                Special cases:

                * -1: Not confident enough for prediction
                * -2: Label not in training
                * -3: Closest cluster > epsilon
        """
        # Run fit and predict sequentially
        return self.fit(
            X          = X,
            y          = y,
            scores     = scores,
            iterations = iterations,
            batch_size = batch_size,
            strategy   = strategy,
            NO_SCORE   = NO_SCORE,
            verbose    = verbose,
        ).predict(
            X          = X,
            y          = y,
            iterations = 100,
            batch_size = 1024,
            verbose    = False,
        )

    ########################################################################
    #                              Clustering                              #
    ########################################################################

    def cluster(self, X, y, iterations=100, batch_size=1024, verbose=False):
        """Cluster contexts in X for same output event y.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context to cluster.

            y : torch.Tensor of shape=(n_samples, 1)
                Events to cluster.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            clusters : np.array of shape=(n_samples,)
                Clusters per input sample.
            """
        ####################################################################
        #                   Represent context as vector                    #
        ####################################################################

        # Get optimized vectors
        #对input进行处理！
        vectors, mask = self.attended_context(
            X                = X,
            y                = y,
            threshold        = self.threshold,
            iterations       = iterations,
            batch_size       = batch_size,
            verbose          = verbose,
        )

        ####################################################################
        #                     Group sequences by event                     #
        ####################################################################

        # Group sequences for clustering per event type

        # print('group by')
        # npX=np.asarray(X)
        # key=lambda x: x.data.tobytes()
        # for index, label in enumerate(npX):
        #     print(label)
        #     print(index)
        #     hashed = key(label)
        #     print('hashed')
        #     print(hashed)
        #     break
            # Add label to lookup table if it does not exist
            # if hashed not in groups:
            #     groups[hashed] = [key(label), list()]
            # # Append item
            # groups[hashed][1].append(index)
        # print(X[0].data.tobytes())



        indices_y = group_by(
            X   = y[mask].squeeze(1).cpu().numpy(),
            key = lambda x: x.data.tobytes(),
        )
        # print("indices_y")
        # print(len(indices_y))
        # for i in range(len(indices_y)):
        #     print(indices_y[i][1].shape)
        # print(X[indices_y[0][1]].shape)
        # print(X[indices_y[0][1]])
        #为什么array不同但是to bytes却是相同的
        # Add verbosity if necessary
        if verbose: indices_y = tqdm(indices_y, desc="Clustering      ")

        ####################################################################
        #                          Cluster events                          #
        ####################################################################

        # Initialise result for confident samples
        result = np.full(mask.sum(), -1, dtype=int)

        # Loop over each event
        for event, context_mask in indices_y:
         
            # Compute clusters per event
            clusters = self.dbscan.dbscan(
                X           = vectors[context_mask],
                eps         = self.eps,
                min_samples = self.min_samples,
                verbose     = False,
            )

            # Add offset to clusters to ensure unique identifiers per event
            clusters[clusters != -1] += max(0, result.max() + 1)#确保不同的cluster里面是不同的数字，因为每一个循环都是不同的类
            
            # Set resulting clusters
            result[context_mask] = clusters

        ####################################################################
        #                    Add non-confident clusters                    #
        ####################################################################

        # Set clusters to -1 by default, i.e., if not confident
        clusters = np.full(mask.shape[0], -1, dtype=int)
        # Add confident clusters
        clusters[mask.cpu().numpy()] = result

        ####################################################################
        #                         Store in object                          #
        ####################################################################

        # Store clusters
        self.clusters = clusters
        # Store vectors
        self.vectors = vectors
        # Store events
        self.events = y.reshape(-1).cpu().numpy()

        # Return clusters
      
        return clusters
        #self.cluster把每一个sequence都赋予了一个数字。代表了聚在第几类
    ########################################################################
    #                            Manual scoring                            #
    ########################################################################

    def score(self, scores, verbose=False):
        """Assigns score to clustered samples.

            Parameters
            ----------
            scores : array-like of shape=(n_samples,)
                Scores of individual samples.

            verbose : boolean, default=False
                If True, print progress.

            Returns
            -------
            self : self
                Returns self
            """
        # Cast scores to numpy array
        scores = np.asarray(scores)

        ################################################################
        #                        Perform checks                        #
        ################################################################

        # Check if scores have same shape as clusters
        if scores.shape != self.clusters.shape:
            raise ValueError(
                "Shape of scores {} did not match shape of clusters {}".format(
                scores.shape,
                self.clusters.shape,
            ))

        # Check if score for each cluster are equal
        for cluster, indices in group_by(self.clusters):
            if np.unique(scores[indices]).shape[0] != 1:
                raise ValueError(
                    "Cluster {} contains different scores. Please use the "
                    "Interpreter.score_clusters function to assign the same "
                    "score to all samples in a cluster.".format(cluster)
                )

        ################################################################
        #                        Assign scores                         #
        ################################################################

        # Retrieve scores for clustered events only
        scores = scores[self.clusters != -1]

        # Compute clustered events
        clustered_events = group_by(self.events[self.clusters != -1])

        # If verbose, add printing
        if verbose: clustered_events = tqdm(clustered_events, desc="Scoring")
        count=0
        # Loop over all clustered events
        for event, indices in clustered_events:#不是按照每一个cluster分的，而是按照每一个event分的
            # Get relevant vectors for given event
            vectors = self.vectors[indices]
            shape=vectors.shape
            # Get unique vectors - optimizes computation time
            vectors, inverse, _ = sp_unique(vectors)

            # Compute KDTree for vectors
            self.tree[event] = KDTree(vectors.toarray(), p=1)

            # Compute scores for given tree indices
            self.labels[event] = dict()
            score = scores[indices]
            data, index_tree, _, _ = self.tree[event].get_arrays()
            _, index_vector = zip(*group_by(inverse))
            assert np.all(data == vectors.toarray())
            if count==0:
                print('shape')
                print(shape)
                print(vectors.shape)
                print(indices)
                print(inverse)
                print('index_vector')
                print(index_vector)
                print(_)
                print(index_tree)
            for index, mapping in zip(index_tree, index_vector):
                # if count==0:
                #     print('mapping')
                #     print(mapping)
                #     print(index)
                    
                self.labels[event][index] = score[mapping].max()
            count+=1
        # Return self
        return self


    def score_clusters(self, scores, strategy="max", NO_SCORE=-1):
        """Compute score per cluster based on individual scores and given
            strategy.

            Parameters
            ----------
            scores : array-like of float, shape=(n_samples,)
                Scores for each sample in cluster.

            strategy : string (max|min|avg), default=max
                Strategy to use for computing scores per cluster based on scores
                of individual events. Currently available options are:
                - max: Use maximum score of any individual event in a cluster.
                - min: Use minimum score of any individual event in a cluster.
                - avg: Use average score of any individual event in a cluster.

            NO_SCORE : float, default=-1
                Score to indicate that no score was given to a sample and that
                the value should be ignored for computing the cluster score.
                The NO_SCORE value will also be given to samples that do not
                belong to a cluster.

            Returns
            -------
            scores : np.array of shape=(n_samples)
                Scores for individual sequences computed using clustering
                strategy. All datapoints within a cluster are guaranteed to have
                the same score.
            """
        # Cast scores to numpy array
        scores = np.asarray(scores)

        # Initialise result
        result = np.full(scores.shape[0], NO_SCORE, dtype=float)#注意这个no_score的值

        # Check if scores are same shape as clusters
        if scores.shape != self.clusters.shape:
            raise ValueError(
                "Scores and stored clusters should have the same shape, but "
                "instead we found '{}' scores and '{}' cluster entries".format(
                scores       .shape,
                self.clusters.shape,
            ))

        # Group by clusters
        #indices是同一个score的所有index位置
        for cluster, indices in group_by(self.clusters):
            # Skip "no cluster" cluster
            if cluster == -1: continue

            # Get relevant scores
            scores_ = scores[indices]
            scores_ = scores_[scores_ != NO_SCORE]

            # Raise error in case scores cannot be computed because of NO_SCORE
            if scores_.shape[0] == 0:
                raise ValueError(
                    "Cannot compute cluster score for cluster '{}'. All "
                    "sequences in this cluster have been assigned score "
                    "NO_SCORE == {}.".format(cluster, NO_SCORE)
                )

            # Apply strategy
            if strategy == "max":
                score = scores_.max()
            elif strategy == "min":
                score = scores_.min()
            elif strategy == "avg":
                score = scores_.mean()
            else:
                raise NotImplementedError(
                    "Unknown strategy: '{}'".format(strategy)
                )

            # Add score to result
            result[indices] = score

        # Return result
        return result


    ########################################################################
    #            Computing total attention per contextual event            #
    ########################################################################

    def vectorize(self, X, attention, size):
        """Compute the total attention for each event in the context.
            The resulting vector can be used to compare sequences.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, sequence_length, input_dim)
                Context events to vectorize.

            attention : torch.Tensor of shape=(n_samples, sequence_length)
                Attention for each event.

            size : int
                Total number of possible events, determines the vector size.

            Returns
            -------
            result : scipy.sparse.csc_matrix of shape=(n_samples, n)
                Sparse vector representing each context.
            """
        # Initialise result
        result = sp.csc_matrix((X.shape[0], size))
        range  = np.arange(X.shape[0], dtype=int)

        # Create vectors
        for i, events in enumerate(torch.unbind(X, dim=1)):
            result += sp.csc_matrix(
                (attention[:, i].detach().cpu().numpy(),
                (range, events.cpu().numpy())),
                shape=(X.shape[0], size)
            )

        # Return result
        return result


    def attended_context(self, X, y,
            threshold  = 0.2,
            iterations = 100,
            batch_size = 1024,
            verbose    = False,
        ):
        """Get vectors representing context after the attention query.

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, seq_length)
                Input context to cluster.

            y : torch.Tensor of shape=(n_samples, 1)
                Events to cluster.

            threshold : float, default=0.2
                Minimum confidence required for creating a vector representing
                the context.

            iterations : int, default=100
                Number of iterations for query.

            batch_size : int, default=1024
                Size of batch for query.

            verbose : boolean, default=False
                If True, prints achieved speedup of clustering algorithm.

            Returns
            -------
            vectors : scipy.sparse.csc_matrix of shape=(n_samples, dim_vector)
                Sparse vectors representing each context with a
                confidence >= threshold.

            mask : np.array of shape=(n_samples,)
                Boolean array of masked vectors. True where input has
                confidence >= threshold, False otherwise.
            """

        ####################################################################
        #                        Optimize attention                        #
        ####################################################################

        logger.info("attended_context: Optimize attention")

        # Get optimal confidence
        confidence, attention = self.attention_query(
            X          = X,
            y          = y,
            iterations = iterations,
            batch_size = batch_size,
            verbose    = verbose,
        )

        # Check where confidence is above threshold
        # print('confidence_inter')
        # print(confidence.shape)
        # print(attention.shape)
        mask = confidence >= threshold
        # print('mask')
        # print(mask.shape)
        # print(X.shape)
        # print(X.shape)
        # print(X[mask].shape)
        # print(attention[mask].shape)
        # print(self.features)
        logger.info("attended_context: Optimize attention finished")

        ####################################################################
        #         Create vectors (total attention for each event)          #
        ####################################################################

        logger.info("attended_context: Create vectors")

        # Perform vectorization
        #把每一个event sequence的vector都扩展到300，聚类的时候就可以对齐了
        vectors = self.vectorize(
            X         = X[mask],
            attention = attention[mask],
            size      = self.features,
        )
        print('vector')
        print(X.shape)
        print(X[mask].shape)
        print(attention[mask].shape)
        print(type(vectors))
        print(vectors.shape)
        print(confidence.shape)
        # Round attention to 4 decimal places (for quicker analysis)
        vectors = np.round(vectors, decimals=4)

        logger.info("attended_context: Create vectors finished")

        ####################################################################
        #                          Return result                           #
        ####################################################################

        # Return result
        return vectors, mask


    ########################################################################
    #                           Attention Query                            #
    ########################################################################

    def attention_query(self, X, y, iterations=100, batch_size=1024, verbose=False):
        """Compute optimal attention for given context X.

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context of events, same as input to fit and predict.

            y : array-like of type=int and shape=(n_samples,)
                Observed event.

            iterations : int, default=100
                Number of iterations to perform for optimization of actual
                event.

            batch_size : int, default=1024
                Batch size of items to optimize.

            verbose : boolean, default=False
                If True, prints progress.

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples,)
                Resulting confidence levels in y.

            attention : torch.Tensor of shape=(n_samples,)
                Optimal attention for predicting event y.
            """
        # Get unique values
        X, y, inverse = unique_2d(X, y)

        # Perform query
        confidence, attention, _ = self.context_builder.query(
            X          = X,
            y          = y,
            iterations = iterations,
            batch_size = batch_size,
            verbose    = verbose,
        )

        # Compute confidence of y
        # print('confidence_attn')
        # print(confidence.shape)
        # print(y.squeeze(1))
        confidence = confidence[torch.arange(y.shape[0]), y.squeeze(1)]

        # Return confidence and attention
        return confidence[inverse], attention[inverse]


    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_dict(self):
        """Return a pickle-compatible dictionary representation of the
            interpreter.

            Returns
            -------
            result : dict()
                JSON-compatible dictionary representation of the (trained)
                interpreter.
            """
        logger.info("to_dict")

        return {
            # Object parameters
            'features'   : self.features,
            'eps'        : self.eps,
            'min_samples': self.min_samples,
            'threshold'  : self.threshold,

            # Stored entries
            'clusters': self.clusters,
            'vectors' : self.vectors,
            'events'  : self.events,

            # Trained features
            'tree'       : self.tree,
            'labels'     : self.labels,
        }

    @classmethod
    def from_dict(cls, dictionary, context_builder=None):
        """Load the interpreter from the given dictionary.

            Parameters
            ----------
            dictionary : dict()
                Dictionary containing state information of the interpreter to
                load.

            context_builder : ContextBuilder, optional
                If given, use the given ContextBuilder for loading the
                Interpreter.

            Returns
            -------
            interpreter : Interpreter
                Interpreter, constructed from dictionary.
            """
        logger.info("from_dict")

        # Set context_builder if given separately
        if context_builder is not None:
            dictionary['context_builder'] = context_builder

        # List of required features
        features = {
            # ContextBuilder
            'context_builder': None,

            # Interpreter parameters
            'features'       : 100,
            'eps'            : 0.1,
            'min_samples'    : 5,
            'threshold'      : 0.2,

            # Stored entries
            'clusters': np.zeros(0),
            'vectors' : np.zeros((0, 100)),
            'events'  : np.zeros(0),
            'tree'           : dict(),
            'labels'         : dict(),
        }

        # Throw warning if dictionary does not contain values
        for feature, default in features.items():

            # Throw warning if feature not available
            if feature not in dictionary:
                # Throw warning
                warnings.warn(
                    "Loading interpreter from dictionary, required feature '{}'"
                    " not in dictionary. Defaulting to default '{}'".format(
                    feature,
                    default
                ))
                # Set default value
                dictionary[feature] = default

        # Create new instance with given features
        result = cls(
            context_builder= dictionary.get('context_builder'),
            features       = dictionary.get('features') ,
            eps            = dictionary.get('eps'),
            min_samples    = dictionary.get('min_samples'),
            threshold      = dictionary.get('threshold'),
        )

        result.clusters = dictionary.get('clusters')
        result.vectors  = dictionary.get('vectors')
        result.events   = dictionary.get('events')
        result.tree     = dictionary.get('tree')
        result.labels   = dictionary.get('labels')

        # Return result
        return result

    def save(self, outfile):
        """Save model to output file.

            Parameters
            ----------
            outfile : string
                File to output model.
            """
        logger.info("save to {}".format(outfile))

        # Save to output file
        with open(outfile, 'wb') as outfile:
            pickle.dump(self.to_dict(), outfile)


    @classmethod
    def load(cls, infile, context_builder=None):
        """Load model from input file.

            Parameters
            ----------
            infile : string
                File from which to load model.

            context_builder : ContextBuilder, optional
                If given, use the given ContextBuilder for loading the
                Interpreter.

            Returns
            -------
            self : self
                Return self.
            """
        logger.info("load from {}".format(infile))

        # Load data
        with open(infile, 'rb') as infile:
            return Interpreter.from_dict(
                dictionary      = pickle.load(infile),
                context_builder = context_builder,
            )
