# Other imports
import numpy as np
import torch

# DeepCASE Imports
from deepcase.preprocessing import Preprocessor
from deepcase               import DeepCASE
from sklearn.metrics import classification_report
from deepcase.context_builder import ContextBuilder
if __name__ == "__main__":
    ########################################################################
    #                             Loading data                             #
    ########################################################################

    # Create preprocessor
    preprocessor = Preprocessor(
        length  = 10,    # 10 events in context
        timeout = 86400, # Ignore events older than 1 day (60*60*24 = 86400 seconds)
    )

    # Load data from file
    # context, events, labels, mapping = preprocessor.csv('/Users/liguangyu/Downloads/DeepCASE-main-2/attack_data_num.csv')
    context, events, labels, mapping = preprocessor.csv('/Users/liguangyu/Downloads/DeepCASE-main-2/attack_data_300000.csv')
    # print(mapping.keys())
    # print(context[:100])
    
    # print(events[:10])
    # print(mapping)
    # table={}
    # for i in range(len(context)):
    #     if labels[i]!=0:
    #         count=len(torch.unique(context[i]))
    #         # if count in table.keys():
    #         #     table[count]+=1
    #         # else:
    #         #     table[count]=1
    #         if count==4:
    #             print(context[i])
    # print(table)







    # In case no labels are provided, set labels to -1
    if labels is None:
        labels = np.full(events.shape[0], -1, dtype=int)

    # Cast to cuda if available
    if torch.cuda.is_available():
        events  = events .to('cuda')
        context = context.to('cuda')

    ########################################################################
    #                            Splitting data                            #
    ########################################################################

    # Split into train and test sets (20:80) by time - assuming events are ordered chronologically
    events_train  = events [:events.shape[0]//5 ]
    events_test   = events [ events.shape[0]//5:]

    context_train = context[:events.shape[0]//5 ]
    context_test  = context[ events.shape[0]//5:]

    labels_train  = labels [:events.shape[0]//5 ]
    labels_test   = labels [ events.shape[0]//5:]


    ########################################################################
    #                       Training ContextBuilder                        #
    ########################################################################

    # Create ContextBuilder
    context_builder = ContextBuilder(
        input_size    =  500,   # Number of input features to expect
        output_size   =  500,   # Same as input size
        hidden_size   = 128,   # Number of nodes in hidden layer, in paper we set this to 128
        max_length    = 10,    # Length of the context, should be same as context in Preprocessor
    )

    # Cast to cuda if available
    if torch.cuda.is_available():
        context_builder = context_builder.to('cuda')

    # Train the ContextBuilder
    context_builder.fit(
        X             = context_train,               # Context to train with
        y             = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
        epochs        = 7,                         # Number of epochs to train with
        batch_size    = 128,                         # Number of samples in each training batch, in paper this was 128
        learning_rate = 0.01,                        # Learning rate to train with, in paper this was 0.01
        verbose       = True,                        # If True, prints progress
    )

    ########################################################################
    #                  Get prediction from ContextBuilder                  #
    ########################################################################

    # Use context builder to predict confidence
    confidence, _ = context_builder.predict(
        X = context_test
    )

    # Get confidence of the next step, seq_len 0 (n_samples, seq_len, output_size)
    confidence = confidence[:, 0]
    # Get confidence from log confidence
    confidence = confidence.exp()
    # Get prediction as maximum confidence
    y_pred = confidence.argmax(dim=1)

    ########################################################################
    #                          Perform evaluation                          #
    ########################################################################

    # Get test and prediction as numpy array
    y_test = events_test.cpu().numpy()
    y_pred = y_pred     .cpu().numpy()

    # Print classification report
    print(classification_report(
        y_true = y_test,
        y_pred = y_pred,
        digits = 4,
    ))













    ########################################################################
    #                            Using DeepCASE                            #
    ########################################################################

    # deepcase = DeepCASE(
    #     # ContextBuilder parameters
    #     features    = 300, # Number of input features to expect
    #     max_length  = 10,  # Length of the context, should be same as context in Preprocessor
    #     hidden_size = 128, # Number of nodes in hidden layer, in paper we set this to 128

    #     # Interpreter parameters
    #     eps         = 0.1, # Epsilon value to use for DBSCAN clustering, in paper this was 0.1
    #     min_samples = 5,   # Minimum number of samples to use for DBSCAN clustering, in paper this was 5
    #     threshold   = 0.2, # Confidence threshold used for determining if attention from the ContextBuilder can be used, in paper this was 0.2
    # )

    # # Cast to cuda if available
    # if torch.cuda.is_available():
    #     deepcase = deepcase.to('cuda')



    # print(events_train.reshape(-1, 1).shape)



    
    # ########################################################################
    # #                             Fit DeepCASE                             #
    # ########################################################################

    # # Train the ContextBuilder
    # # Conveniently, the fit and fit_predict methods have the same API, so if you
    # # do not require the predicted values on the train dataset, simply
    # # substitute fit_predict with fit and it will run slightly quicker because
    # # DeepCASE skip the prediction over the training dataset and simply return
    # # the deepcase object itself. Other than that, both calls are exactly the
    # # same.
    # prediction_train = deepcase.fit_predict(
    #     # Input data
    #     X      = context_train,               # Context to train with
    #     y      = events_train.reshape(-1, 1), # Events to train with, note that these should be of shape=(n_events, 1)
    #     scores = labels_train,                # Labels used to compute score (either as loaded by Preprocessor, or put your own labels here)

    #     # ContextBuilder-specific parameters
    #     epochs        = 100,                   # Number of epochs to train with
    #     batch_size    = 128,                  # Number of samples in each training batch, in paper this was 128
    #     learning_rate = 0.01,                 # Learning rate to train with, in paper this was 0.01

    #     # Interpreter-specific parameters
    #     iterations       = 100,               # Number of iterations to use for attention query, in paper this was 100
    #     query_batch_size = 1024,              # Batch size to use for attention query, used to limit CUDA memory usage
    #     strategy         = "max",             # Strategy to use for scoring (one of "max", "min", "avg")
    #     NO_SCORE         = -1,                # Any sequence with this score will be ignored in the strategy.
    #                                           # If assigned a cluster, the sequence will inherit the cluster score.
    #                                           # If the sequence is not present in a cluster, it will receive a score of NO_SCORE.

    #     # Verbosity level
    #     verbose = True,                       # If True, prints progress
    # )
    # print(prediction_train [:1000])
    # print(np.where(prediction_train >0))
    # print(context_test.shape)
    # print(events_test.shape)
    # print(np.where(events_train.reshape(-1, 1)>0))
    # print((labels_train>0).sum())
    # ########################################################################
    # #                        Predict with DeepCASE                         #
    # ########################################################################

    # # # Compute predicted scores
    # # prediction_test = deepcase.predict(
    # #     X          = context_test,               # Context to predict
    # #     y          = events_test.reshape(-1, 1), # Events to predict, note that these should be of shape=(n_events, 1)
    # #     iterations = 100,                        # Number of iterations to use for attention query, in paper this was 100
    # #     batch_size = 1024,                       # Batch size to use for attention query, used to limit CUDA memory usage
    # #     verbose    = True,                       # If True, prints progress
    # # )
    # # print(prediction_test[:1000])
    # # print(np.where(prediction_test>0))
    # # print(context_test.shape)
    # # print(events_test.shape)

    # # confidence, _ = deepcase.context_builder.predict(
    # #     X          = context_test,               # Context to predict
    # #   )
    # #  # Get confidence of the next step, seq_len 0 (n_samples, seq_len, output_size)
    # # confidence = confidence[:, 0]
    # # # Get confidence from log confidence
    # # confidence = confidence.exp()
    # # # Get prediction as maximum confidence
    # # y_pred = confidence.argmax(dim=1)

    # # ########################################################################
    # # #                          Perform evaluation                          #
    # # ########################################################################

    # # # Get test and prediction as numpy array
    # # y_test = events_test.cpu().numpy()
    # # y_pred = y_pred     .cpu().numpy()
    # # print(len(y_test))
    # # num=len(np.where(y_test==y_pred))
    # # print(num/len(y_test))
    # # # print()
    # # # Print classification report
    # # print(classification_report(
    # #     y_true = y_test,
    # #     y_pred = y_pred,
    # #     digits = 4,
    # # ))
   