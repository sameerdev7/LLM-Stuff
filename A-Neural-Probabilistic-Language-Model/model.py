import torch
import torch.nn as nn
import torch.nn.functional as F

class TrigramModel:
    """Extended version of Karpathy's model with trigrams for better text generation"""
    
    def __init__(self, text_file='shakespeare.txt', embedding_dim=10, context_size=3):
        # Read text
        with open(text_file, 'r') as f:
            self.text = f.read()
        
        # Build vocabulary
        chars = sorted(list(set(self.text)))
        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Text length: {len(self.text)} characters")
        
        # Context size (bigram=1, trigram=2, etc.)
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        
        # Build dataset
        self.xs, self.ys = self.build_dataset()
        self.num = len(self.xs)
        
        print(f"Training examples: {self.num}")
        
        # Model parameters
        self.C = nn.Embedding(self.vocab_size, embedding_dim)  # Character embeddings
        self.W = torch.randn((embedding_dim * context_size, self.vocab_size), requires_grad=True)
        self.b = torch.randn(self.vocab_size, requires_grad=True)
    
    def build_dataset(self):
        """Build training dataset with context windows"""
        xs, ys = [], []
        
        # Add start tokens
        context = [0] * self.context_size
        
        for ch in self.text[:100000]:  # Limit for memory - adjust as needed
            # Skip characters not in vocabulary
            if ch not in self.stoi:
                continue
            
            ix = self.stoi[ch]
            xs.append(context[:])
            ys.append(ix)
            context = context[1:] + [ix]
        
        return torch.tensor(xs), torch.tensor(ys)
    
    def train(self, num_iterations=1000, learning_rate=0.1, batch_size=1024):
        """Train the model with mini-batch gradient descent"""
        for k in range(num_iterations):
            # Mini-batch sampling
            ix = torch.randint(0, self.num, (batch_size,))
            
            # Forward pass
            emb = self.C(self.xs[ix])  # (batch_size, context_size, embedding_dim)
            emb = emb.view(emb.shape[0], -1)  # Flatten: (batch_size, context_size * embedding_dim)
            logits = emb @ self.W + self.b  # (batch_size, vocab_size)
            
            # Loss
            loss = F.cross_entropy(logits, self.ys[ix])
            
            # Backward pass
            for p in [self.C.weight, self.W, self.b]:
                if p.grad is not None:
                    p.grad.zero_()
            
            loss.backward()
            
            # Update with learning rate decay
            lr = learning_rate * (0.99 ** (k // 100))
            with torch.no_grad():
                self.C.weight -= lr * self.C.weight.grad
                self.W -= lr * self.W.grad
                self.b -= lr * self.b.grad
            
            if k % 100 == 0:
                print(f"Iteration {k}: loss = {loss.item():.4f}, lr = {lr:.6f}")
    
    def generate_text(self, seed_text="", max_length=500, temperature=1.0):
        """Generate text starting from seed"""
        g = torch.Generator().manual_seed(42)
        
        # Initialize context
        if seed_text:
            context = [self.stoi.get(ch, 0) for ch in seed_text[-self.context_size:]]
            # Pad if needed
            while len(context) < self.context_size:
                context = [0] + context
            out = list(seed_text)
        else:
            context = [0] * self.context_size
            out = []
        
        # Generate
        for _ in range(max_length):
            # Forward pass
            emb = self.C(torch.tensor([context]))
            emb = emb.view(1, -1)
            logits = emb @ self.W + self.b
            
            # Apply temperature
            probs = F.softmax(logits / temperature, dim=1)
            
            # Sample
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            
            # Stop at end token (optional)
            if ix == 0 and len(out) > 10:
                break
            
            # Add character
            if ix != 0:
                out.append(self.itos[ix])
            
            # Update context
            context = context[1:] + [ix]
        
        return ''.join(out)


# Usage
if __name__ == "__main__":
    # Initialize model
    print("Initializing model...")
    model = TrigramModel(
        text_file='shakespeare.txt', 
        embedding_dim=10, 
        context_size=3
    )
    
    # Train
    print("\nTraining...")
    model.train(num_iterations=2000, learning_rate=0.1, batch_size=2048)
    
    # Generate samples
    print("\n" + "="*60)
    print("GENERATED TEXT")
    print("="*60)
    
    seeds = ["ROMEO: ", "JULIET: ", "To be or not to be", ""]
    for seed in seeds:
        print(f"\n--- Seed: '{seed}' ---")
        text = model.generate_text(seed_text=seed, max_length=300, temperature=0.8)
        print(text)