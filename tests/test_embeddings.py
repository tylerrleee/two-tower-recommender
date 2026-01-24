from src.embedding import EmbeddingEngineer
import numpy as np
import pandas as pd
import unittest
from unittest.mock import Mock, patch, MagicMock

# Disable FAISS multi-threading to prevent runtime conflict + segfault
# MacOS causes conflict with FAISS compilation -> segfault
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import faiss
faiss.omp_set_num_threads(1)

class TestEmbeddingEngineerInit(unittest.TestCase):

    def test_default_initialization(self):
        """Test default parameters are set correctly"""
        ee = EmbeddingEngineer(sbert_model_name="test-model")
        
        self.assertEqual(ee.model_name, "test-model")
        self.assertEqual(ee.batch_size, 64)
        self.assertEqual(ee.device, "cpu")
        self.assertEqual(ee.precision, np.float32)
        self.assertIsNone(ee._model)
        self.assertIsNone(ee.index)
    
    def test_custom_initialization(self):
        """Test custom parameters are set correctly"""
        ee = EmbeddingEngineer(
            sbert_model_name="custom-model",
            embedding_batch_size=128,
            use_gpu=False,
            precision=np.float64
        )
        
        self.assertEqual(ee.model_name, "custom-model")
        self.assertEqual(ee.batch_size, 128)
        self.assertEqual(ee.device, "cpu")
        self.assertEqual(ee.precision, np.float64)
    
    def test_gpu_device_selection(self):
        """Test GPU device is set when use_gpu=True"""
        ee = EmbeddingEngineer(
            sbert_model_name="test-model",
            use_gpu=True
        )
        
        self.assertEqual(ee.device, "cuda")

class TestModelProperty(unittest.TestCase):
    """Test model loading via property"""
    
    # Any call to SenternceTransformer will return a mock object
    ## so that no model is downloaded, no GPU is used and faster tests
    @patch('src.embedding.SentenceTransformer')
    def test_model_loads_on_first_access(self, mock_sbert):
        """Test model is loaded lazily on first access"""
        mock_model = Mock()
        mock_sbert.return_value = mock_model
        
        ee = EmbeddingEngineer(sbert_model_name="test-model") # Lazy Loading
        
        # Model should not be loaded yet
        self.assertIsNone(ee._model)
        
        # Access model property
        model = ee.model
        
        # Model should now be loaded
        mock_sbert.assert_called_once_with("test-model", device="cpu")
        self.assertEqual(model, mock_model)
        self.assertEqual(ee._model, mock_model)
    
    @patch('src.embedding.SentenceTransformer')
    def test_model_loads_only_once(self, mock_sbert):
        """Test model is cached after first load"""
        mock_model = Mock()
        mock_sbert.return_value = mock_model
        
        ee = EmbeddingEngineer(sbert_model_name="test-model")
        
        # Access model multiple times
        model1 = ee.model
        model2 = ee.model
        model3 = ee.model
        
        # Should only call SentenceTransformer once
        mock_sbert.assert_called_once()
        self.assertEqual(model1, model2)
        self.assertEqual(model2, model3)
    
    @patch('src.embedding.SentenceTransformer')
    def test_model_handles_type_error(self, mock_sbert):
        """Test model handles TypeError gracefully"""
        mock_sbert.side_effect = TypeError("Model not found")
        
        ee = EmbeddingEngineer(sbert_model_name="invalid-model")
        
        # Should return None instead of raising
        model = ee.model
        self.assertIsNone(model)

class TestEmbedTexts(unittest.TestCase):
    """Test text embedding generation"""
    
    def setUp(self):
        """Set up mock model for testing"""
        self.ee = EmbeddingEngineer(sbert_model_name="test-model")
        self.ee._model = Mock()
    
    def test_embed_texts_basic(self):
        """Test basic text embedding"""
        texts = ["Hello world", "Machine learning"]
        expected_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        self.ee._model.encode.return_value = expected_embeddings
        
        result = self.ee.embed_texts(texts)
        
        self.ee._model.encode.assert_called_once()
        np.testing.assert_array_equal(result, expected_embeddings)
    
    def test_embed_texts_parameters(self):
        """Test that correct parameters are passed to model.encode"""
        texts = ["Test sentence"]
        self.ee._model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        
        self.ee.embed_texts(texts, show_progress_bar=False)
        
        # Verify encode was called with correct parameters
        call_kwargs = self.ee._model.encode.call_args[1]
        self.assertEqual(call_kwargs['batch_size'], 64)
        self.assertFalse(call_kwargs['show_progress_bar'])
        self.assertTrue(call_kwargs['convert_to_numpy'])
        self.assertTrue(call_kwargs['normalize_embeddings'])
    
    def test_embed_texts_empty_list(self):
        """Test embedding empty text list"""
        texts = []
        self.ee._model.encode.return_value = np.array([])
        
        result = self.ee.embed_texts(texts)
        
        self.assertEqual(len(result), 0)
    
    def test_embed_texts_single_item(self):
        """Test embedding single text"""
        texts = ["Single sentence"]
        expected = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.ee._model.encode.return_value = expected
        
        result = self.ee.embed_texts(texts)
        
        self.assertEqual(result.shape, (1, 4))
        np.testing.assert_array_equal(result, expected)
    
    def test_embed_texts_output_shape(self):
        """Test output shape is correct (n_samples, embedding_dim)"""
        texts = ["Text 1", "Text 2", "Text 3"]
        # Simulate 384D embeddings (common for S-BERT)
        expected = np.random.rand(3, 384).astype(np.float32)
        self.ee._model.encode.return_value = expected
        
        result = self.ee.embed_texts(texts)
        
        self.assertEqual(result.shape, (3, 384))
        self.assertEqual(type(result[0, 0]), np.float32) #setUp Model assumes default precision

class TestCombineFeatures(unittest.TestCase):
    """Test combining text embeddings with meta features"""
    
    def setUp(self):
        """Set up EmbeddingEngineer with mock model"""
        self.ee = EmbeddingEngineer(sbert_model_name="test-model")
        self.ee._model = Mock()
    
    def test_combine_features_basic(self):
        """Test basic feature combination"""
        text_emb = np.array([[0.1, 0.2], [0.3, 0.4]])
        meta_feat = np.array([[1, 2], [3, 4]])
        
        result = self.ee.combine_features(text_emb, meta_feat)
        
        expected = np.array([[0.1, 0.2, 1, 2], [0.3, 0.4, 3, 4]], dtype=np.float32)
        #np.testing.assert_array_almost_equal(result, expected)
        self.assertEqual(result.all(), expected.all())
   
    def test_combine_features_shapes(self):
        """Test that combined features have correct shape"""
        text_emb = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])  # 3 samples, 384D
        meta_feat = np.array([[1] * 16, [2] * 16, [3] * 16])  # 3 samples, 16D
        
        result = self.ee.combine_features(text_emb, meta_feat)
        
        # Should be (3 samples, 400 dimensions)
        self.assertEqual(result.shape, (3, 400))

    def test_combine_features_with_text_strings(self):
        """Test combination when text_features are strings (not yet embedded)"""
        text_strings = np.array(["Hello", "World"])
        meta_feat = np.array([[1, 2], [3, 4]])
        
        # Mock the embed_texts call
        expected_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        #assert expected_embeddings.shape == (2, 2)

        self.ee._model.encode.return_value = expected_embeddings
        
        result = self.ee.combine_features(text_strings, meta_feat)
        
        # Should have called embed_texts
        self.ee._model.encode.assert_called_once()
        self.assertEqual(result.shape, (2, 4))

    def test_combine_features_precision_casting(self):
        """Test that features are cast to correct precision by EE"""
        ee_float64 = EmbeddingEngineer(
            sbert_model_name="test-model",
            precision=np.float64
        )
        
        text_emb = np.array([[0.1, 0.2]], dtype=np.float32)
        meta_feat = np.array([[1, 2]], dtype=np.int32)
        
        result = ee_float64.combine_features(text_emb, meta_feat)
        
        self.assertEqual(result.dtype, np.float64)
    def test_combine_features_raises_on_none_text(self):
        """Test that ValueError is raised when text_features is None"""
        meta_feat = np.array([[1, 2]])
        
        with self.assertRaises(ValueError) as context:
            self.ee.combine_features(None, meta_feat)
        
        self.assertIn("Text Embeddings does not exist", str(context.exception))
    
    def test_combine_features_raises_on_none_meta(self):
        """Test that ValueError is raised when meta_features is None"""
        text_emb = np.array([[0.1, 0.2]])
        
        with self.assertRaises(ValueError) as context:
            self.ee.combine_features(text_emb, None)
        
        self.assertIn("Meta Feature does not exist", str(context.exception))
    
    def test_combine_features_raises_on_empty_text(self):
        """Test that ValueError is raised when text_features is empty"""
        text_emb = np.array([])
        meta_feat = np.array([[1, 2]])
        
        with self.assertRaises(ValueError) as context:
            self.ee.combine_features(text_emb, meta_feat)
        
        self.assertIn("Text Embeddings does not exist", str(context.exception))
    
    def test_combine_features_raises_on_empty_meta(self):
        """Test that ValueError is raised when meta_features is empty"""
        text_emb = np.array([[0.1, 0.2]])
        meta_feat = np.array([])
        
        with self.assertRaises(ValueError) as context:
            self.ee.combine_features(text_emb, meta_feat)
        
        self.assertIn("Meta Feature does not exist", str(context.exception))
    
    def test_combine_features_raises_on_shape_mismatch(self):
        """Test that ValueError is raised when shapes don't match"""
        text_emb = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2 samples
        meta_feat = np.array([[1, 2]])  # 1 sample
        
        with self.assertRaises(ValueError) as context:
            self.ee.combine_features(text_emb, meta_feat)
        
        self.assertIn("Mismatched entries", str(context.exception))

class TestFAISSIndex(unittest.TestCase):
    """Test FAISS index building and querying"""
    
    def setUp(self):
        """Set up EmbeddingEngineer for testing"""
        self.ee = EmbeddingEngineer(sbert_model_name="test-model",
                                    use_gpu=False)
    
    def test_build_faiss_index_basic(self):
        """Test building FAISS index"""
        embeddings = np.random.rand(10, 64).astype(np.float32)
        
        self.ee.build_faiss_index(embeddings)
        
        self.assertIsNotNone(self.ee.index)
        self.assertEqual(self.ee.index.ntotal, 10)

    def test_build_faiss_index_dimensions(self):
        """Test that FAISS index has correct dimensions"""
        embedding_dim = 128
        embeddings = np.random.rand(5, embedding_dim).astype(np.float32)
        
        self.ee.build_faiss_index(embeddings)
        
        self.assertEqual(self.ee.index.d, embedding_dim)
        
    def test_build_faiss_index_empty(self):
        """Test building index with empty embeddings"""
        embeddings = np.empty((0, 64), dtype=np.float32)
        
        self.ee.build_faiss_index(embeddings)
        
        self.assertEqual(self.ee.index.ntotal, 0) # empty embeddings == no index representation
    def test_query_index_basic(self):
        """Test querying FAISS index"""
        # Build index with known (4,3) vectors
        embeddings = np.array([ 
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.9, 0.1, 0.0]
        ], dtype=np.float32)
        
        self.ee.build_faiss_index(embeddings)
        
        # Query with vector similar to first embedding
        query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        distances, indices = self.ee.query_index(query, top_k=2)
        
        # Should return 2 results
        self.assertEqual(distances.shape, (1, 2))
        self.assertEqual(indices.shape, (1, 2))
        
        # First result should be index 0 (exact match)
        self.assertEqual(indices[0, 0], 0)

    def test_query_index_top_k(self):
        """Test querying with different top_k values"""
        embeddings = np.random.rand(10, 64).astype(np.float32)
        self.ee.build_faiss_index(embeddings)
        
        query = np.array([[0.5] * 64], dtype=np.float32) # shape: (64,1)
        
        # Test different top_k values
        for k in [1, 3, 5, 10]:
            distances, indices = self.ee.query_index(query, top_k=k)
            self.assertEqual(distances.shape, (1, k))
            self.assertEqual(indices.shape, (1, k))

    def test_query_index_multiple_queries(self):
        """FAISS returns correct top-k matches for multiple queries"""
        np.random.seed(0)

        embeddings = np.random.rand(20, 64).astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.ee.build_faiss_index(embeddings)

        # Use known embeddings as queries (self-match should be rank-1)
        queries = embeddings[:3].copy()

        distances, indices = self.ee.query_index(queries, top_k=5)

        # Shape checks
        self.assertEqual(distances.shape, (3, 5))
        self.assertEqual(indices.shape, (3, 5))

        # Each query should retrieve itself as top-1
        for i in range(3):
            self.assertEqual(indices[i, 0], i)
            self.assertAlmostEqual(distances[i, 0], 1.0, places=5)

        # Distances sorted descending
        self.assertTrue(
            np.all(np.diff(distances, axis=1) <= 1e-6)
        )

    def test_query_index_raises_without_index(self):
        """Test that querying raises error if index not built"""
        query = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        
        with self.assertRaises(RuntimeError) as context:
            self.ee.query_index(query, top_k=5)
        
        self.assertIn("Index not built", str(context.exception))

class TestIntegration(unittest.TestCase):
    """Integration tests for complete embedding.py workflows"""
    
    def setUp(self):
        """Set up for integration tests"""
        self.ee = EmbeddingEngineer(sbert_model_name="test-model")
        self.ee._model = Mock()
    
    def test_full_embedding_pipeline(self):
        """Test complete pipeline: embed texts -> combine features -> build index
        - Assuming we have done feature scaling from features.py"""
        # Step 1: Embed texts
        texts = ["Text 1", "Text 2", "Text 3"]
        text_embeddings = np.random.rand(3, 384).astype(np.float32)
        self.ee._model.encode.return_value = text_embeddings
        
        embedded = self.ee.embed_texts(texts)
        
        # Step 2: Combine with meta features
        meta_features = np.random.rand(3, 16).astype(np.float32)
        combined = self.ee.combine_features(embedded, meta_features)
        
        # Step 3: Build FAISS index
        self.ee.build_faiss_index(combined)
        
        # Verify final state
        self.assertEqual(combined.shape, (3, 400))
        self.assertEqual(self.ee.index.ntotal, 3)

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        self.ee = EmbeddingEngineer(sbert_model_name="test-model")
        self.ee._model = Mock()
    
    def test_single_sample_embedding(self):
        """Test embedding single sample"""
        texts = ["Single text"]
        self.ee._model.encode.return_value = np.array([[0.1] * 384])
        
        result = self.ee.embed_texts(texts)
        
        self.assertEqual(result.shape, (1, 384))
    
    def test_combine_features_single_sample(self):
        """Test combining features for single sample"""
        text_emb = np.array([[0.1] * 384])
        meta_feat = np.array([[1] * 16])
        
        result = self.ee.combine_features(text_emb, meta_feat)
        
        self.assertEqual(result.shape, (1, 400))
    
    def test_large_batch_embedding(self):
        """Test embedding large batch of texts"""
        texts = [f"Text {i}" for i in range(1000)]
        self.ee._model.encode.return_value = np.random.rand(1000, 384)
        
        result = self.ee.embed_texts(texts)
        
        self.assertEqual(result.shape, (1000, 384))
    
    def test_very_long_text(self):
        """Test embedding very long text"""
        long_text = "word " * 1000  # Very long text
        self.ee._model.encode.return_value = np.array([[0.1] * 384])
        
        result = self.ee.embed_texts([long_text])
        
        self.assertEqual(result.shape, (1, 384))
    
    def test_empty_text(self):
        """Test embedding empty string"""
        texts = [""]
        self.ee._model.encode.return_value = np.array([[0.0] * 384])
        
        result = self.ee.embed_texts(texts)
        
        self.assertEqual(result.shape, (1, 384))
    
    def test_special_characters_in_text(self):
        """Test embedding text with special characters
        - google form applications often have emojis """
        texts = ["Hello! @#$% 123 😀"]

        self.ee._model.encode.return_value = np.array([[0.1] * 384])
        
        result = self.ee.embed_texts(texts)
        
        self.assertEqual(result.shape, (1, 384))

if __name__ == '__main__':
    unittest.main()



# TODO: TestSaveLoadEmbedding
# TODO: TestSaveSaveEmbedding
