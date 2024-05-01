import numpy as np
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
from typing import List, Tuple, Dict
from matplotlib import pyplot as plt
from rdkit.Chem import MolFromSmiles
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate, LSTM, Bidirectional, Dense, Input, GaussianNoise, BatchNormalization, RepeatVector, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, History, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from my_Vocabulary import Vocabulary

class Autoencoder:
    def __init__(self, model_path: str, input_shape: Tuple[int, int], latent_dim: int, lstm_units: int, output_dim: int, batch_norm: bool, batch_norm_momentum: float, noise_std: float, numb_dec_layer: int, emb_dim: int, vocab_size: int, max_len: int):
        """
        Initializes the Autoencoder class to build different components of the SMILES to latent space and back to SMILES model.

        Args:
        - model_path (str): Directory path where the model and related data are stored or will be saved.
        - input_shape (Tuple[int, int]): Shape of the input data, typically (max_len, vocab_size).
        - latent_dim (int): Dimensionality of the latent space.
        - lstm_units (int): Number of units in LSTM layers.
        - output_dim (int): Dimensionality of the model's output, usually the same as vocab_size.
        - batch_norm (bool): Whether to use batch normalization.
        - batch_norm_momentum (float): Momentum for the batch normalization layers.
        - noise_std (float): Standard deviation of Gaussian noise to add to the input layer.
        - numb_dec_layer (int): Number of decoder layers.
        - emb_dim (int): Dimensionality of the embedding layer.
        - vocab_size (int): Size of the vocabulary.
        - max_len (int): Maximum length of input sequences.

        Builds:
        - SMILES to latent space model.
        - Latent to LSTM states model.
        - LSTM states to SMILES model.
        - Full model connecting all parts.
        """
        self.path = model_path
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.output_dim = output_dim
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.noise_std = noise_std
        self.numb_dec_layer = numb_dec_layer
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Constructing various components of the autoencoder
        self.build_smiles_to_latent_model()
        self.build_latent_to_states_model()
        self.build_states_to_smiles_model()

        # Building the full autoencoder model
        self.build_model()

        # Print the model summary to verify the construction
        print(self.model.summary())
        
    def build_smiles_to_latent_model(self):
        """
        Constructs the part of the autoencoder that encodes SMILES strings to their latent representations.
        This includes multiple layers of bidirectional LSTMs to process the sequences both forwards and backwards.
        """
        # Input layer accepts variable-length sequences
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        x = Embedding(self.vocab_size, self.lstm_units // 2)(encoder_inputs)

        # Lists to store forward and reverse states
        states_list: List[tf.Tensor] = []
        states_reversed_list: List[tf.Tensor] = []

        # Construct bidirectional LSTM layers
        for i in range(self.numb_dec_layer):
            return_sequences = True if i != self.numb_dec_layer - 1 else False
            encoder = Bidirectional(LSTM(self.lstm_units // 2, return_sequences=return_sequences,
                                         return_state=True, name=f'encoder{i}_LSTM'))

            outputs = encoder(x)
            x = outputs[0]  # LSTM output
            states = outputs[1:]  # LSTM states

            # Collect LSTM states
            states_list.extend(states[:2])  # forward states
            states_reversed_list.extend(states[2:])  # reverse states

            # Batch normalization after each LSTM layer except the last
            if self.batch_norm and i != self.numb_dec_layer - 1:
                x = BatchNormalization(momentum=self.batch_norm_momentum, name=f'BN_{i}')(x)

        # Combine all collected states into one vector
        complete_states_list = states_list + states_reversed_list
        states = Concatenate(axis=-1, name='concatenate')(complete_states_list)

        # Apply batch normalization to the combined states
        if self.batch_norm:
            states = BatchNormalization(momentum=self.batch_norm_momentum, name=f'BN_{self.numb_dec_layer}')(states)

        # Dense layer to produce the latent representation
        latent_representation = Dense(self.latent_dim, activation="relu", name="Dense_relu_latent_rep")(states)
        if self.batch_norm:
            latent_representation = BatchNormalization(momentum=self.batch_norm_momentum, name='BN_latent_rep')(latent_representation)

        # Add Gaussian Noise to the latent representation as a regularization technique
        latent_representation = GaussianNoise(self.noise_std, name='Gaussian_Noise')(latent_representation)

        # Create the model object
        self.smiles_to_latent_model = Model(encoder_inputs, latent_representation, name='smiles_to_latent_model')

        # Save the model summary to a file
        with open('smiles_to_latent.txt', 'w') as f:
            self.smiles_to_latent_model.summary(print_fn=lambda x: f.write(x + '\n'))
            
    def build_latent_to_states_model(self):
        """
        Constructs the model that, given a latent representation, reconstructs the initial states (h and c)
        for each layer of a LSTM decoder.
        """
        latent_input = Input(shape=(self.latent_dim,), name='latent_input')

        # List to hold the reconstructed LSTM states
        decoded_states: List[tf.Tensor] = []

        # Build dense layers to reconstruct LSTM states for each decoder layer
        for dec_layer in range(self.numb_dec_layer):  # Typically, 2 layers as per the comment
            # Constructing the 'h' state decoder
            h_decoder = Dense(self.lstm_units, activation="relu", name=f"Dense_h_{dec_layer}")(latent_input)
            # Constructing the 'c' state decoder
            c_decoder = Dense(self.lstm_units, activation="relu", name=f"Dense_c_{dec_layer}")(latent_input)

            # Optional batch normalization
            if self.batch_norm:
                h_decoder = BatchNormalization(momentum=self.batch_norm_momentum, name=f"BN_h_{dec_layer}")(h_decoder)
                c_decoder = BatchNormalization(momentum=self.batch_norm_momentum, name=f"BN_c_{dec_layer}")(c_decoder)

            # Append the reconstructed states to the list
            decoded_states.append(h_decoder)
            decoded_states.append(c_decoder)

        # Create the model object
        self.latent_to_states_model = Model(latent_input, decoded_states, name='latent_to_states_model')

        # Save the model summary to a file for inspection
        with open('latent_to_states.txt', 'w') as f:
            self.latent_to_states_model.summary(print_fn=lambda x: f.write(x + '\n'))
            
    def build_states_to_smiles_model(self):
        """
        Builds a model that, given LSTM decoder states, reconstructs the SMILES sequence.
        This involves setting up LSTM layers initialized with specific states and applying a dense layer to generate character probabilities.
        """
        # Decoder inputs for teacher forcing during training
        decoder_inputs = Input(shape=self.input_shape, name="decoder_inputs")

        # Collect all model inputs: decoder inputs and reconstructed states
        inputs = [decoder_inputs]
        x = decoder_inputs

        # Adding LSTM layers initialized with reconstructed states
        for dec_layer in range(self.numb_dec_layer):
            state_h_input = Input(shape=(self.lstm_units,), name=f"Decoded_state_h_{dec_layer}")
            state_c_input = Input(shape=(self.lstm_units,), name=f"Decoded_state_c_{dec_layer}")
            inputs.extend([state_h_input, state_c_input])

            # Setup LSTM layer with initial states
            decoder_lstm = LSTM(self.lstm_units, return_sequences=True, name=f"Decoder_LSTM_{dec_layer}")
            x = decoder_lstm(x, initial_state=[state_h_input, state_c_input])

            # Optional batch normalization
            if self.batch_norm:
                x = BatchNormalization(momentum=self.batch_norm_momentum, name=f"BN_decoder_{dec_layer}")(x)

        # Final Dense layer to output probabilities for each SMILES character
        outputs = Dense(self.output_dim, activation="softmax", name="Decoder_Dense")(x)

        # Creating the model
        self.states_to_smiles_model = Model(inputs=inputs, outputs=outputs, name="states_to_smiles_model")

        # Writing model summary to a file for inspection
        with open('states_to_smiles.txt', 'w') as f:
            self.states_to_smiles_model.summary(print_fn=lambda x: f.write(x + '\n'))
            
    def build_model(self):
        """
        Constructs the complete autoencoder model that links the encoding of SMILES to their latent representation
        and decodes them back to SMILES using teacher forcing.
        """
        # Inputs for the encoder and decoder parts of the model
        encoder_inputs = Input(shape=(None,), name="encoder_inputs")
        decoder_inputs = Input(shape=self.input_shape, name="decoder_inputs")

        # Process encoder inputs through the encoder model to obtain latent representation
        latent_representation = self.smiles_to_latent_model(encoder_inputs)

        # Process latent representation through the model that constructs initial decoder states
        initial_states = self.latent_to_states_model(latent_representation)

        # Combine decoder inputs and initial states for the decoding process
        decoder_outputs = self.states_to_smiles_model([decoder_inputs] + initial_states)

        # Creating the full autoencoder model
        self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name="Autoencoder")

        # Print model summary to inspect the architecture
        self.model.summary()
        
    def load_autoencoder_model(self, path: str):
        """
        Loads the weights for the autoencoder model from the specified path and rebuilds the sampling model.
        """
        self.model.load_weights(path)
        self.build_sample_model()
        
    def fit_model(self, dataX, dataX2, dataY, epochs: int, batch_size: int, optimizer: str):
        """
        Fits the model to the data.

        Args:
        - dataX: Input data for the encoder.
        - dataX2: Additional input data for the decoder (used in teacher forcing).
        - dataY: Target data.
        - epochs (int): Number of epochs to train.
        - batch_size (int): Batch size for training.
        - optimizer (str): Optimizer type, 'adam' or 'adam_clip'.

        Sets up the optimizer, compiles the model, executes the training, and plots training history.
        """
        self.epochs = epochs
        self.batch_size = batch_size

        # Configure optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate=0.001)
        elif optimizer == 'adam_clip':
            self.optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False, clipvalue=3)

        # Setting up callbacks
        checkpoint_dir = self.path
        checkpoint_file = f"{checkpoint_dir}model--{{epoch:02d}}--{{val_loss:.4f}}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_file, monitor="val_loss", mode="min", save_best_only=True)
        #reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

        callbacks_list = [checkpoint]#, reduce_lr]

        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy')

        # Execute training
        results = self.model.fit(
            [dataX, dataX2], dataY, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.1, 
            shuffle=True, 
            verbose=1, 
            callbacks=callbacks_list
        )

        # Plot training history
        fig, ax = plt.subplots()
        ax.plot(results.history['loss'], label="Train")
        ax.plot(results.history['val_loss'], label="Val")
        ax.legend()
        ax.set(xlabel='Epochs', ylabel='Loss')
        figure_path = f"{self.path}Loss_plot_{dataX.shape[0]}.png"
        fig.savefig(figure_path)
        plt.close(fig)  # Close the plot to free up resources

        # Build sampling model post-training
        self.build_sample_model()
        
    def build_sample_model(self):
        """
        Configures a sampling model by modifying the states_to_smiles_model to be stateful and have an adaptable input shape.
        This function explicitly handles each step in the process, making modifications and adjustments to the configuration of the model.
        """
        # Retrieve the original configuration of the states_to_smiles_model
        config = self.states_to_smiles_model.get_config()

        # Modify the input layers configuration to keep only the "Decoder_Inputs"
        config["input_layers"] = [config["input_layers"][0]]

        # Identify and remove layers that are specific to 'decoded states'
        decoded_state_indices = []
        for idx, layer in enumerate(config["layers"]):
            if "Decoded_state_" in layer["name"]:
                decoded_state_indices.append(idx)

        # Remove inbound_nodes dependencies of remaining layers on deleted ones
        for layer in config["layers"]:
            if "inbound_nodes" in layer and layer["inbound_nodes"]:
                idx_list = []
                # Attempt to iterate over the first list of inbound_nodes
                try:
                    for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                        # Check if the inbound node is iterable and if it contains a reference to a 'Decoded_state_' layer
                        if isinstance(inbound_node, (list, tuple)) and any("Decoded_state_" in n[0] for n in inbound_node if isinstance(n, (list, tuple))):
                            idx_list.append(idx)
                # Catch the exception if layer["inbound_nodes"][0] is unexpectedly structured or empty
                except Exception as e:
                    print("Error processing inbound nodes for layer:", layer["name"], "; Error:", e)
                    continue

                # Remove indexed inbound nodes from the list, reverting indices to avoid re-arranging
                for idx in sorted(idx_list, reverse=True):
                    layer["inbound_nodes"][0].pop(idx)

        # Change the batch shape of the input layer to allow for stateful operations
        config["layers"][0]["config"]["batch_input_shape"] = (1, 1, self.output_dim)

        # Set LSTM layers to be stateful
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True

        # Rebuild the model from the modified configuration
        sample_model = Model.from_config(config)

        # Copy weights from the trained model to the new sample model
        for layer in sample_model.layers:
            original_layer = self.states_to_smiles_model.get_layer(layer.name)
            if original_layer is not None:
                layer.set_weights(original_layer.get_weights())

        # Store the newly created sample model
        self.sample_model = sample_model
        return config  # Returning the config could be useful for debugging or further modification

    def latent_to_smiles(self, latent: np.ndarray, vocab) -> str:
        """
        Generates a SMILES string from a latent representation using a trained sample model.

        Args:
        latent (np.ndarray): Latent representation of a SMILES string.
        vocab (Vocabulary): Vocabulary object containing mapping and other helper functions.

        Returns:
        str: A SMILES string predicted from the latent representation.
        """
        # Predict the initial states from the latent representation
        states = self.latent_to_states_model.predict(latent)

        # Reset states in the sample model for each decoder layer
        for dec_layer in range(self.numb_dec_layer):
            self.sample_model.get_layer(f"Decoder_LSTM_{dec_layer}").reset_states(
                states=[states[2 * dec_layer], states[2 * dec_layer + 1]]
            )

        # Initialize the sample vector with the start token
        sample_vector = np.zeros((1, 1, vocab.vocab_size))
        sample_vector[0, 0, vocab.char_to_int["G"]] = 1

        # Generate characters until max length or stop character is reached
        smiles = ""
        for _ in range(vocab.max_len):
            pred = self.sample_model.predict(sample_vector)
            idx = np.argmax(pred)
            char = vocab.int_to_char[idx]
            if char != "A":
                smiles += char
                sample_vector = np.zeros((1, 1, vocab.vocab_size))
                sample_vector[0, 0, idx] = 1
            else:
                break

        # Convert any placeholder tokens to actual chemical symbols
        smiles = vocab.replace_tokens_by_atoms(smiles)
        return smiles

    def evaluate_reconstruction(self, real: List[str], predicted: List[str]) -> float:
        """
        Compares real and predicted SMILES strings to calculate the accuracy of reconstruction.

        Args:
        real (List[str]): Original SMILES strings.
        predicted (List[str]): Predicted SMILES strings.

        Returns:
        float: Percentage of accurately reconstructed SMILES strings.
        """
        assert len(real) == len(predicted)
        correct = sum(1 for real_sm, pred_sm in zip(real, predicted) if real_sm == pred_sm)
        return correct / len(real) * 100

    def validity(self, smiles_list: List[str]) -> Tuple[List[str], float]:
        """
        Evaluates the validity of generated SMILES strings using RDKit.

        Args:
        smiles_list (List[str]): List of SMILES strings.

        Returns:
        Tuple[List[str], float]: A tuple containing a list of valid SMILES strings and the percentage of valid strings.
        """
        valid_smiles = [sm for sm in smiles_list if MolFromSmiles(sm)]
        perc_valid = len(valid_smiles) / len(smiles_list) * 100 if smiles_list else 0
        return valid_smiles, perc_valid
    
if __name__ == "__main__":
    # Path setup and data loading
    file_path_vae_train = r'C:\Users\mokht\Desktop\GAN-DRUG-GENERATOR\my_src\clean_smiles.csv'
    vocab_file_path = r'C:\Users\mokht\Desktop\GAN-Drug-Generator\my_src\vocab.txt'
    df_vae_train = pd.read_csv(file_path_vae_train)
    print(df_vae_train.head())
    
    smiles = df_vae_train['SMILES'].tolist()
    smiles = smiles[:10000]

    # Initialize the Vocabulary
    vocab = Vocabulary(vocab_file_path, max_len=100)
    vocab.update_vocab(smiles)

    # Tokenize and prepare data
    tok_smiles, _ = vocab.tokenize(smiles)
    tok_train = tok_smiles[1000:]  # Assume the first 1000 are for testing
    tok_test = tok_smiles[:1000]

    # Encode and prepare datasets
    X_train = vocab.one_hot_encoder(tok_train)
    Y_train = vocab.get_target(X_train, encode='OHE')

    # Define model parameters
    latent_dim = 256
    lstm_units = 512
    epochs = 50
    batch_size = 128
    batch_norm = True
    batch_norm_momentum = 0.9
    numb_dec_layer = 2
    noise_std = 0.1
    input_shape = X_train.shape[1:]
    output_dim = vocab.tokenizer.get_vocab_size()

    # Initialize the Autoencoder
    auto = Autoencoder(
        model_path='',
        input_shape=input_shape,
        latent_dim=latent_dim,
        lstm_units=lstm_units,
        output_dim=output_dim,
        batch_norm=batch_norm,
        batch_norm_momentum=batch_norm_momentum,
        noise_std=noise_std,
        numb_dec_layer=numb_dec_layer,
        emb_dim=output_dim,
        vocab_size=output_dim,
        max_len=vocab.max_len
    )

    # Optionally, compile and train the model
    auto.fit_model(X_train, X_train, Y_train, epochs, batch_size, 'adam')

    # Save the autoencoder summary to a file
    with open('Summary_autoencoder.txt', 'w') as fh:
        auto.model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print(auto.sample_model.summary())
