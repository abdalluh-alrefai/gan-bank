import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from collections import Counter
import os

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import plot_model

np.random.seed(42)
tf.random.set_seed(42)


print("\nStep 1: Data Loading and Exploration")
print("="*50)

try:
    df = pd.read_csv('bank.csv')
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Check for missing values
print("\nMissing values check:")
print(df.isnull().sum())

# Explore the target variable distribution
print("\nTarget variable distribution:")
print(df['deposit'].value_counts())

# Visualize class imbalance
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='deposit')
plt.title('Class Distribution of Deposit')
plt.show()

# Step 2: Data Preprocessing
print("\n\nStep 2: Data Preprocessing")
print("="*50)

# Separate features and target
X = df.drop('deposit', axis=1)
y = df['deposit'].map({'yes': 1, 'no': 0})  # Convert to binary

# Identify categorical and numerical columns
cat_cols = [col for col in X.columns if X[col].dtype == 'object']
num_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ])

# Split data before preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Check the shape after preprocessing
print(f"\nPreprocessed training data shape: {X_train_preprocessed.shape}")
print(f"Preprocessed test data shape: {X_test_preprocessed.shape}")

# Get minority class samples (deposit = yes)
minority_indices = np.where(y_train == 1)[0]
X_minority = X_train_preprocessed[minority_indices]
print(f"\nMinority class samples shape: {X_minority.shape}")

# Step 3: GAN Implementations
print("\n\nStep 3: GAN Implementations")
print("="*50)

def build_vanilla_gan(input_dim, latent_dim=100):
    """Build a Vanilla GAN model"""
    # Generator
    generator = models.Sequential(name="Generator")
    generator.add(layers.Input(shape=(latent_dim,)))
    generator.add(layers.Dense(128, activation='relu'))
    generator.add(layers.Dense(256, activation='relu'))
    generator.add(layers.Dense(512, activation='relu'))
    generator.add(layers.Dense(input_dim, activation='tanh'))
    
    # Discriminator
    discriminator = models.Sequential(name="Discriminator")
    discriminator.add(layers.Input(shape=(input_dim,)))
    discriminator.add(layers.Dense(512, activation='relu'))
    discriminator.add(layers.Dense(256, activation='relu'))
    discriminator.add(layers.Dense(128, activation='relu'))
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile discriminator
    discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
    
    # Combined GAN model
    gan = models.Sequential(name="VanillaGAN")
    gan.add(generator)
    discriminator.trainable = False
    gan.add(discriminator)
    
    gan.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
               loss='binary_crossentropy')
    
    return generator, discriminator, gan

def build_dcgan(input_dim, latent_dim=100):
    """Build a DCGAN model"""
    # Generator
    generator = models.Sequential(name="DCGAN_Generator")
    generator.add(layers.Input(shape=(latent_dim,)))
    generator.add(layers.Dense(128, activation='relu'))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(256, activation='relu'))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(512, activation='relu'))
    generator.add(layers.BatchNormalization())
    generator.add(layers.Dense(input_dim, activation='tanh'))
    
    # Discriminator with dropout
    discriminator = models.Sequential(name="DCGAN_Discriminator")
    discriminator.add(layers.Input(shape=(input_dim,)))
    discriminator.add(layers.Dense(512, activation='relu'))
    discriminator.add(layers.Dropout(0.3))
    discriminator.add(layers.Dense(256, activation='relu'))
    discriminator.add(layers.Dropout(0.3))
    discriminator.add(layers.Dense(128, activation='relu'))
    discriminator.add(layers.Dropout(0.3))
    discriminator.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile discriminator
    discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
    
    # Combined GAN model
    gan = models.Sequential(name="DCGAN")
    gan.add(generator)
    discriminator.trainable = False
    gan.add(discriminator)
    
    gan.compile(optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
               loss='binary_crossentropy')
    
    return generator, discriminator, gan

def train_gan(generator, discriminator, gan, X_train, latent_dim, epochs, batch_size, sample_interval):
    """Train GAN model"""
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Training history
    d_losses = []
    g_losses = []
    d_accs = []
    
    for epoch in range(epochs):
        
        # Select a random batch of real samples
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]
        
        # Generate a batch of fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise, verbose=0)
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_samples, valid)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
       
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)
        
        # Store losses and accuracies
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        d_accs.append(d_loss[1])
        
        # Print progress
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(d_accs, label='Discriminator Accuracy', color='green')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return generator

# Parameters (adjusted for stability)
input_dim = X_train_preprocessed.shape[1]
latent_dim = 100
batch_size = 32
epochs = 500  # Reduced for demonstration
sample_interval = 50

# Check if GPU is available
print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")

# Build and train Vanilla GAN
try:
    print("\nBuilding and training Vanilla GAN...")
    vanilla_generator, vanilla_discriminator, vanilla_gan = build_vanilla_gan(input_dim, latent_dim)
    print("\nVanilla GAN summary:")
    vanilla_generator.summary()
    vanilla_discriminator.summary()
    
    vanilla_generator = train_gan(vanilla_generator, vanilla_discriminator, vanilla_gan, 
                                X_minority, latent_dim, epochs, batch_size, sample_interval)
except Exception as e:
    print(f"\nError training Vanilla GAN: {e}")
    print("Trying with smaller batch size...")
    try:
        batch_size = 16
        vanilla_generator = train_gan(vanilla_generator, vanilla_discriminator, vanilla_gan, 
                                    X_minority, latent_dim, epochs, batch_size, sample_interval)
    except Exception as e:
        print(f"\nStill failing with error: {e}")
        print("Consider using CPU-only mode or reducing model complexity.")

# Build and train DCGAN
try:
    print("\nBuilding and training DCGAN...")
    dcgan_generator, dcgan_discriminator, dcgan_gan = build_dcgan(input_dim, latent_dim)
    print("\nDCGAN summary:")
    dcgan_generator.summary()
    dcgan_discriminator.summary()
    
    dcgan_generator = train_gan(dcgan_generator, dcgan_discriminator, dcgan_gan, 
                              X_minority, latent_dim, epochs, batch_size, sample_interval)
except Exception as e:
    print(f"\nError training DCGAN: {e}")
    print("Trying with smaller batch size...")
    try:
        batch_size = 16
        dcgan_generator = train_gan(dcgan_generator, dcgan_discriminator, dcgan_gan, 
                                  X_minority, latent_dim, epochs, batch_size, sample_interval)
    except Exception as e:
        print(f"\nStill failing with error: {e}")
        print("Consider using CPU-only mode or reducing model complexity.")

# Step 4: Generate Synthetic Samples and Balance Dataset
print("\n\nStep 4: Generate Synthetic Samples and Balance Dataset")
print("="*50)

def generate_samples(generator, latent_dim, n_samples):
    """Generate synthetic samples using trained generator"""
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_samples = generator.predict(noise)
    return generated_samples

# Calculate how many samples we need to balance the dataset
n_minority = sum(y_train == 1)
n_majority = sum(y_train == 0)
n_samples_needed = n_majority - n_minority

# Generate synthetic samples with both GANs
vanilla_generated = generate_samples(vanilla_generator, latent_dim, n_samples_needed)
dcgan_generated = generate_samples(dcgan_generator, latent_dim, n_samples_needed)


# Vanilla GAN augmented
X_train_vanilla = np.vstack([X_train_preprocessed, vanilla_generated])
y_train_vanilla = np.concatenate([y_train, np.ones(n_samples_needed)])

# DCGAN augmented
X_train_dcgan = np.vstack([X_train_preprocessed, dcgan_generated])
y_train_dcgan = np.concatenate([y_train, np.ones(n_samples_needed)])

# Original imbalanced data
X_train_original = X_train_preprocessed
y_train_original = y_train

# Check the new class distributions
print("\nClass distributions:")
print("Original:", Counter(y_train_original))
print("Vanilla GAN augmented:", Counter(y_train_vanilla))
print("DCGAN augmented:", Counter(y_train_dcgan))

# Step 5: Classification and Evaluation
print("\n\nStep 5: Classification and Evaluation")
print("="*50)

def train_and_evaluate(X_train, y_train, X_test, y_test, name):
    """Train classifier and evaluate performance"""
    print(f"\nTraining and evaluating with {name} data...")
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return metrics

# Evaluate on original imbalanced data
original_metrics = train_and_evaluate(X_train_original, y_train_original, X_test_preprocessed, y_test, "Original Imbalanced")

# Evaluate on Vanilla GAN augmented data
vanilla_metrics = train_and_evaluate(X_train_vanilla, y_train_vanilla, X_test_preprocessed, y_test, "Vanilla GAN Augmented")

# Evaluate on DCGAN augmented data
dcgan_metrics = train_and_evaluate(X_train_dcgan, y_train_dcgan, X_test_preprocessed, y_test, "DCGAN Augmented")

# Compare results
metrics_df = pd.DataFrame([original_metrics, vanilla_metrics, dcgan_metrics],
                         index=['Original', 'Vanilla GAN', 'DCGAN'])

print("\nPerformance Comparison:")
print(metrics_df)

# Plot metrics comparison
plt.figure(figsize=(12,6))
metrics_df.plot(kind='bar', rot=0)
plt.title('Classifier Performance Comparison')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Step 6: Visualizing Generated Samples
print("\n\nStep 6: Visualizing Generated Samples")
print("="*50)

# Apply PCA for visualization
pca = PCA(n_components=2)
original_pca = pca.fit_transform(X_train_preprocessed)
vanilla_pca = pca.transform(vanilla_generated)
dcgan_pca = pca.transform(dcgan_generated)

# Plot the results
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.scatter(original_pca[y_train==0,0], original_pca[y_train==0,1], alpha=0.5, label='Majority (no)')
plt.scatter(original_pca[y_train==1,0], original_pca[y_train==1,1], alpha=0.5, label='Minority (yes)')
plt.title('Original Data Distribution')
plt.legend()

plt.subplot(1,3,2)
plt.scatter(original_pca[:,0], original_pca[:,1], alpha=0.2, label='Original')
plt.scatter(vanilla_pca[:,0], vanilla_pca[:,1], alpha=0.5, c='red', label='Vanilla GAN Generated')
plt.title('Vanilla GAN Generated Samples')
plt.legend()

plt.subplot(1,3,3)
plt.scatter(original_pca[:,0], original_pca[:,1], alpha=0.2, label='Original')
plt.scatter(dcgan_pca[:,0], dcgan_pca[:,1], alpha=0.5, c='green', label='DCGAN Generated')
plt.title('DCGAN Generated Samples')
plt.legend()

plt.tight_layout()
plt.show()

# Step 7: Conclusion and Analysis
print("\n\nStep 7: Conclusion and Analysis")
print("="*50)

print("\nFinal Performance Comparison:")
print(metrics_df)

# Convert dictionary values to numpy arrays for calculations
original_values = np.array(list(original_metrics.values()))
vanilla_values = np.array(list(vanilla_metrics.values()))
dcgan_values = np.array(list(dcgan_metrics.values()))

# Calculate improvement over original
improvement = pd.DataFrame({
    'Vanilla GAN Improvement (%)': (vanilla_values - original_values) / original_values * 100,
    'DCGAN Improvement (%)': (dcgan_values - original_values) / original_values * 100
}, index=original_metrics.keys())

print("\nImprovement over Original Imbalanced Data (%):")
print(improvement)

# Plot improvement
plt.figure(figsize=(12,6))
improvement.plot(kind='bar', rot=0)
plt.title('Percentage Improvement over Original Imbalanced Data')
plt.ylabel('Improvement (%)')
plt.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.show()

print("\nProject completed successfully!")