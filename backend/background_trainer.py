"""
Background Real-Time Retraining System
Runs in separate thread - does NOT block API
Uses a lightweight CNN for adversarial detection
"""

import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLE CNN MODEL FOR ADVERSARIAL DETECTION
# ============================================================================

class AdversarialDetectionCNN(nn.Module):
    """
    Lightweight CNN for detecting adversarial images
    Binary classification: 0 = clean, 1 = adversarial
    """

    def __init__(self):
        super(AdversarialDetectionCNN, self).__init__()

        # Input: (batch, 3, 32, 32)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (batch, 16, 16, 16)

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (batch, 32, 8, 8)

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (batch, 64, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # Binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

    def predict_risk(self, image_array):
        """
        Predict adversarial risk score (0-100%)
        Input: numpy array of shape (32, 32, 3) or tensor
        Output: risk score (0-100)
        """
        self.eval()

        # Convert numpy array to tensor if needed
        if isinstance(image_array, np.ndarray):
            # Normalize if needed
            if image_array.max() > 1.0:
                image_array = image_array / 255.0

            # Convert to tensor
            image_tensor = torch.from_numpy(image_array).float()
            # Convert from (H, W, 3) to (1, 3, H, W)
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image_array
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

        # Ensure tensor is on correct device
        image_tensor = image_tensor.to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self.forward(image_tensor)
            probs = torch.softmax(logits, dim=1)
            # Class 1 is adversarial, return probability in [0, 1]
            adversarial_prob = probs[0, 1].item()
            return float(adversarial_prob)


class AdversarialImageBuffer:
    """Simple buffer for storing detected adversarial images"""

    def __init__(self, max_size=100):
        self.max_size = max_size
        self.images = []
        self.risk_scores = []
        self.lock = threading.Lock()

    def add(self, image_array, risk_score):
        """Add image to buffer (image_array is numpy 0-1 normalized)"""
        with self.lock:
            if len(self.images) >= self.max_size:
                self.images.pop(0)
                self.risk_scores.pop(0)

            self.images.append(image_array)
            self.risk_scores.append(risk_score)

    def get_and_clear(self):
        """Get all images and clear buffer"""
        with self.lock:
            if len(self.images) == 0:
                return None, None

            images = np.array(self.images)  # Shape: (N, 32, 32, 3)
            risk_scores = np.array(self.risk_scores)

            self.images.clear()
            self.risk_scores.clear()

            return images, risk_scores

    def size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.images)


class BackgroundTrainer(threading.Thread):
    """
    Background thread for real model retraining
    - Waits for buffer to fill
    - Fine-tunes CNN model on adversarial examples
    - Updates weights in main model
    - Does NOT block API
    """

    def __init__(self, model=None, device=None, buffer=None, buffer_threshold=5):
        threading.Thread.__init__(self)
        self.daemon = True  # Background thread

        # Create model if not provided
        if model is None:
            self.model = AdversarialDetectionCNN()
        else:
            self.model = model

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self.model.to(self.device)

        self.buffer = buffer
        self.buffer_threshold = buffer_threshold
        self.running = True
        self.is_retraining = False
        self.retrain_count = 0
        self.last_accuracy = None  # None until actual retraining occurs
        self.lock = threading.Lock()

        logger.info(f"[BackgroundTrainer] Initialized with {self.device}")
        logger.info(f"[BackgroundTrainer] Model: {self.model.__class__.__name__}")

    def run(self):
        """Main background training loop"""
        logger.info("[BackgroundTrainer] Started")

        while self.running:
            # Check if buffer is full enough to retrain
            if self.buffer.size() >= self.buffer_threshold:
                self._do_retrain()

            # Sleep briefly to avoid busy waiting
            threading.Event().wait(0.5)

        logger.info("[BackgroundTrainer] Stopped")

    def _do_retrain(self):
        """Perform actual model retraining with real training loop"""
        with self.lock:
            if self.is_retraining:
                return  # Already retraining

            self.is_retraining = True

        try:
            # Get buffered adversarial images
            images, risk_scores = self.buffer.get_and_clear()

            if images is None or len(images) == 0:
                self.is_retraining = False
                return

            logger.info(f"\n[RETRAIN] Starting with {len(images)} adversarial examples")

            # Convert numpy to torch tensor on GPU
            # Images are (N, 32, 32, 3) in range [0, 1]
            images_tensor = torch.from_numpy(images).float()  # Shape: (N, 32, 32, 3)

            # Normalize and rearrange to (N, 3, 32, 32)
            images_tensor = images_tensor.permute(0, 3, 1, 2)  # (N, 3, 32, 32)
            images_tensor = images_tensor.to(self.device)

            # Generate labels: higher risk = more adversarial
            # Use continuous risk scores for better training
            labels = torch.tensor(
                [1 if score > 50 else 0 for score in risk_scores],
                dtype=torch.long,
                device=self.device
            )

            # Create dataset and dataloader
            dataset = TensorDataset(images_tensor, labels)
            batch_size = min(4, len(images))  # Small batch for small buffer
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Setup training
            self.model.train()
            criterion = nn.CrossEntropyLoss()
            # Very low learning rate for fine-tuning
            optimizer = optim.Adam(self.model.parameters(), lr=0.00001)

            # Fine-tune for 3 epochs
            num_epochs = 3
            total_loss = 0.0
            num_correct = 0
            total_samples = 0

            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_correct = 0

                for batch_images, batch_labels in loader:
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(batch_images)
                    loss = criterion(outputs, batch_labels)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    epoch_correct += (predicted == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)

                avg_loss = epoch_loss / len(loader)
                total_loss += avg_loss
                epoch_acc = (epoch_correct / total_samples * 100) if total_samples > 0 else 0
                logger.info(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}, Acc = {epoch_acc:.1f}%")

            # Record genuine training metrics
            final_loss = total_loss / num_epochs if num_epochs > 0 else 0.0
            self.retrain_count += 1
            self.last_loss = float(final_loss)
            self.last_accuracy = float(epoch_acc)

            logger.info(
                f"[RETRAIN] Complete! Retrain #{self.retrain_count}, "
                f"Trained on: {len(risk_scores)} adversarial samples, "
                f"Final Avg Loss: {final_loss:.6f}, Batch Acc: {epoch_acc:.1f}%\n"
            )

        except Exception as e:
            logger.error(f"[RETRAIN ERROR] {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            with self.lock:
                self.is_retraining = False

    def stop(self):
        """Stop the background trainer"""
        self.running = False

    def get_status(self):
        """Get retraining status"""
        with self.lock:
            return {
                'is_retraining': self.is_retraining,
                'retrain_count': self.retrain_count,
                'current_accuracy': self.last_accuracy,
                'last_loss': getattr(self, 'last_loss', 0.0)
            }
