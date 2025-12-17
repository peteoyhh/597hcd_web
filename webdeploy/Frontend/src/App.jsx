import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import { API_BASE_URL, CATEGORIES } from './config';
import { validateTitle, validateHashtags, validateDuration, sanitizeErrorMessage, isProduction } from './utils/validation';

function App() {
  const [formData, setFormData] = useState({
    title: '',
    hashtags: '',
    category: 'Comedy',
    category_id: 0,
    duration_sec: 0,
    has_description: 0,
  });

  const [computedFields, setComputedFields] = useState({
    title_length: 0,
    hashtag_count: 0,
    log_duration: 0,
  });

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // YouTube crawl and batch prediction states
  const [youtubeApiKey, setYoutubeApiKey] = useState('');
  const [crawledVideos, setCrawledVideos] = useState(null);
  const [crawlLoading, setCrawlLoading] = useState(false);
  const [crawlError, setCrawlError] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState(null);

  // Auto-compute fields when inputs change
  React.useEffect(() => {
    const titleLength = formData.title.trim() ? formData.title.trim().split(/\s+/).length : 0;
    const hashtagCount = formData.hashtags.trim() ? formData.hashtags.trim().split(/\s+/).length : 0;
    const logDuration = formData.duration_sec > 0 ? Math.log1p(formData.duration_sec) : 0;

    setComputedFields({
      title_length: titleLength,
      hashtag_count: hashtagCount,
      log_duration: logDuration,
    });
  }, [formData.title, formData.hashtags, formData.duration_sec]);

  // Update category_id when category changes
  React.useEffect(() => {
    const selectedCategory = CATEGORIES.find(cat => cat.name === formData.category);
    if (selectedCategory) {
      setFormData(prev => ({ ...prev, category_id: selectedCategory.id }));
    }
  }, [formData.category]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    // Apply validation and sanitization for text inputs
    if (name === 'title') {
      const validation = validateTitle(value);
      if (validation.valid) {
        setFormData(prev => ({ ...prev, [name]: validation.sanitized }));
      } else {
        // Still allow typing, but store the value (validation happens on submit)
        setFormData(prev => ({ ...prev, [name]: value }));
      }
    } else if (name === 'hashtags') {
      const validation = validateHashtags(value);
      if (validation.valid) {
        setFormData(prev => ({ ...prev, [name]: validation.sanitized }));
      } else {
        setFormData(prev => ({ ...prev, [name]: value }));
      }
    } else if (name === 'duration_sec') {
      const validation = validateDuration(value);
      if (validation.valid) {
        setFormData(prev => ({ ...prev, [name]: validation.sanitized }));
      } else {
        setFormData(prev => ({ ...prev, [name]: value }));
      }
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: type === 'checkbox' ? (checked ? 1 : 0) : value,
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);

    // Validate inputs before submission
    const titleValidation = validateTitle(formData.title);
    if (!titleValidation.valid) {
      setError(titleValidation.error);
      setLoading(false);
      return;
    }

    const hashtagsValidation = validateHashtags(formData.hashtags);
    if (!hashtagsValidation.valid) {
      setError(hashtagsValidation.error);
      setLoading(false);
      return;
    }

    const durationValidation = validateDuration(formData.duration_sec);
    if (!durationValidation.valid) {
      setError(durationValidation.error);
      setLoading(false);
      return;
    }

    try {
      const payload = {
        ...formData,
        title: titleValidation.sanitized,
        hashtags: hashtagsValidation.sanitized,
        duration_sec: durationValidation.sanitized,
        ...computedFields,
      };

      const response = await axios.post(`${API_BASE_URL}/predict`, payload);
      setResults(response.data);
    } catch (err) {
      const errorMessage = sanitizeErrorMessage(
        err.response?.data?.detail || err.message
      );
      setError(errorMessage);
      if (!isProduction()) {
        console.error('Prediction error:', err);
      }
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return Math.round(num).toLocaleString();
  };

  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
  };

  // Handle YouTube crawl
  const handleCrawlYouTube = async () => {
    const trimmedKey = youtubeApiKey.trim();
    if (!trimmedKey) {
      setCrawlError('Please enter YouTube API key');
      return;
    }

    setCrawlLoading(true);
    setCrawlError(null);
    setCrawledVideos(null);
    setBatchResults(null);

    try {
      // Use POST body instead of URL parameter for security
      const response = await axios.post(
        `${API_BASE_URL}/crawl-youtube`,
        { api_key: trimmedKey },
        { headers: { 'Content-Type': 'application/json' } }
      );
      if (response.data.success) {
        setCrawledVideos(response.data.videos);
      } else {
        const errorMessage = sanitizeErrorMessage(response.data.error);
        setCrawlError(errorMessage || 'Failed to crawl YouTube videos');
      }
    } catch (err) {
      const errorMessage = sanitizeErrorMessage(
        err.response?.data?.detail || err.message
      );
      setCrawlError(errorMessage || 'An error occurred while crawling');
      if (!isProduction()) {
        console.error('Crawl error:', err);
      }
    } finally {
      setCrawlLoading(false);
    }
  };

  // Handle batch prediction
  const handleBatchPredict = async () => {
    if (!crawledVideos || crawledVideos.length === 0) {
      setBatchError('Please crawl YouTube videos first');
      return;
    }

    setBatchLoading(true);
    setBatchError(null);
    setBatchResults(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/batch-predict`, {
        videos: crawledVideos
      });
      if (response.data.success) {
        setBatchResults(response.data.results);
      } else {
        setBatchError('Failed to get predictions');
      }
    } catch (err) {
      const errorMessage = sanitizeErrorMessage(
        err.response?.data?.detail || err.message
      );
      setBatchError(errorMessage || 'An error occurred during prediction');
      if (!isProduction()) {
        console.error('Batch prediction error:', err);
      }
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🎬 YouTube Popularity Predictor</h1>
          <p>Predict video popularity and view growth using AI</p>
        </header>

        <div className="main-content">
          {/* YouTube Crawl and Batch Prediction Section */}
          <div className="batch-section">
            <h2>📊 Batch Analysis</h2>
            <p className="section-description">
              Crawl YouTube videos (10 categories, 3 videos each) and get batch predictions
            </p>
            
            <div className="form-group">
              <label htmlFor="youtube-api-key">YouTube API Key *</label>
              <input
                type="password"
                id="youtube-api-key"
                value={youtubeApiKey}
                onChange={(e) => setYoutubeApiKey(e.target.value)}
                placeholder="Enter your YouTube Data API v3 key"
                className="api-key-input"
              />
              <span className="helper-text">
                Get your API key from <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener noreferrer">Google Cloud Console</a>
              </span>
            </div>
            
            <div className="button-group">
              <button
                type="button"
                onClick={handleCrawlYouTube}
                className="action-btn crawl-btn"
                disabled={crawlLoading}
              >
                {crawlLoading ? '🔄 Crawling...' : '📥 Crawl YouTube Videos'}
              </button>
              
              <button
                type="button"
                onClick={handleBatchPredict}
                className="action-btn predict-btn"
                disabled={batchLoading || !crawledVideos || crawledVideos.length === 0}
              >
                {batchLoading ? '🔮 Predicting...' : '🔮 Batch Predict'}
              </button>
            </div>

            {crawlError && (
              <div className="error-message">
                <strong>Error:</strong> {crawlError}
              </div>
            )}

            {batchError && (
              <div className="error-message">
                <strong>Error:</strong> {batchError}
              </div>
            )}

            {crawledVideos && (
              <div className="crawled-info">
                <p>✅ Crawled {crawledVideos.length} videos from YouTube</p>
              </div>
            )}

            {batchResults && (
              <div className="batch-results">
                <h3>📈 Batch Prediction Results</h3>
                <div className="batch-results-grid">
                  {batchResults.map((result, idx) => (
                    <div key={idx} className="batch-result-card">
                      <div className="result-header">
                        <h4>{result.title || 'Unknown Title'}</h4>
                        <span className="video-id">ID: {result.video_id}</span>
                      </div>
                      
                      <div className="result-category">
                        Category: {result.category}
                      </div>

                      <div className="result-published-date">
                        Published: {result.published_at ? new Date(result.published_at).toLocaleDateString('en-US', { 
                          year: 'numeric', 
                          month: 'short', 
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit'
                        }) : 'N/A'}
                      </div>

                      {result.error ? (
                        <div className="result-error">Error: {result.error}</div>
                      ) : (
                        <>
                          <div className="result-comparison">
                            <div className="comparison-item">
                              <span className="comparison-label">Actual Views:</span>
                              <span className="comparison-value actual">
                                {formatNumber(result.actual_views)}
                              </span>
                            </div>
                            <div className="comparison-item">
                              <span className="comparison-label">Predicted Popularity:</span>
                              <span className={`popularity-badge-small ${result.predicted_popularity.toLowerCase()}`}>
                                {result.predicted_popularity}
                              </span>
                            </div>
                          </div>

                          <div className="predicted-views">
                            <div className="predicted-view-item">
                              <span>1d:</span>
                              <strong>{formatNumber(result.predicted_views.views_1d)}</strong>
                            </div>
                            <div className="predicted-view-item">
                              <span>7d:</span>
                              <strong>{formatNumber(result.predicted_views.views_7d)}</strong>
                            </div>
                            <div className="predicted-view-item">
                              <span>30d:</span>
                              <strong>{formatNumber(result.predicted_views.views_30d)}</strong>
                            </div>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="divider"></div>

          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="form-section">
              <h2>Video Information</h2>
              
              <div className="form-group">
                <label htmlFor="title">Title *</label>
                <input
                  type="text"
                  id="title"
                  name="title"
                  value={formData.title}
                  onChange={handleChange}
                  placeholder="Enter video title"
                  maxLength={200}
                  required
                />
                <span className="helper-text">
                  Words: {computedFields.title_length}
                </span>
              </div>

              <div className="form-group">
                <label htmlFor="hashtags">Hashtags</label>
                <input
                  type="text"
                  id="hashtags"
                  name="hashtags"
                  value={formData.hashtags}
                  onChange={handleChange}
                  placeholder="#youtube #vlog #fun"
                  maxLength={500}
                />
                <span className="helper-text">
                  Count: {computedFields.hashtag_count}
                </span>
              </div>

              <div className="form-group">
                <label htmlFor="category">Category *</label>
                <select
                  id="category"
                  name="category"
                  value={formData.category}
                  onChange={handleChange}
                  required
                >
                  {CATEGORIES.map(cat => (
                    <option key={cat.id} value={cat.name}>
                      {cat.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="duration_sec">Duration (seconds) *</label>
                <input
                  type="number"
                  id="duration_sec"
                  name="duration_sec"
                  value={formData.duration_sec}
                  onChange={handleChange}
                  min="0"
                  max="86400"
                  step="1"
                  placeholder="300"
                  required
                />
                <span className="helper-text">
                  {formData.duration_sec > 0 ? formatDuration(formData.duration_sec) : '0s'} • 
                  Log: {computedFields.log_duration.toFixed(3)}
                </span>
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="has_description"
                    checked={formData.has_description === 1}
                    onChange={handleChange}
                  />
                  Has Description
                </label>
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? 'Predicting...' : '🔮 Predict Popularity'}
            </button>
          </form>

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}

          {results && (
            <div className="results">
              <h2>Prediction Results</h2>
              
              <div className="result-section popularity-section">
                <h3>Popularity Classification</h3>
                <div className={`popularity-badge ${results.popularity_label.toLowerCase()}`}>
                  {results.popularity_label}
                </div>
                <div className="probabilities">
                  <div className="prob-bar">
                    <div className="prob-label">Low</div>
                    <div className="prob-value">
                      {(results.popularity_probabilities.Low * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="prob-bar">
                    <div className="prob-label">Medium</div>
                    <div className="prob-value">
                      {(results.popularity_probabilities.Medium * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="prob-bar">
                    <div className="prob-label">High</div>
                    <div className="prob-value">
                      {(results.popularity_probabilities.High * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              <div className="result-section growth-section">
                <h3>View Growth Prediction</h3>
                <div className="growth-cards">
                  <div className="growth-card">
                    <div className="growth-label">1 Day</div>
                    <div className="growth-value">
                      {formatNumber(results.growth_prediction.views_1d)}
                    </div>
                    <div className="growth-subtitle">views</div>
                  </div>
                  <div className="growth-card">
                    <div className="growth-label">7 Days</div>
                    <div className="growth-value">
                      {formatNumber(results.growth_prediction.views_7d)}
                    </div>
                    <div className="growth-subtitle">views</div>
                  </div>
                  <div className="growth-card">
                    <div className="growth-label">30 Days</div>
                    <div className="growth-value">
                      {formatNumber(results.growth_prediction.views_30d)}
                    </div>
                    <div className="growth-subtitle">views</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

