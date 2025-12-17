// API Configuration
const DEFAULT_API_URL = 'https://597hcdweb-production.up.railway.app';

/**
 * Get and validate API base URL
 * @returns {string} - Validated API URL
 */
const getApiBaseUrl = () => {
  const url = process.env.REACT_APP_API_URL || DEFAULT_API_URL;
  
  // Validate URL format
  try {
    const urlObj = new URL(url);
    // Ensure it's http or https
    if (urlObj.protocol !== 'http:' && urlObj.protocol !== 'https:') {
      console.warn('Invalid API URL protocol, using default');
      return DEFAULT_API_URL;
    }
    return url;
  } catch (error) {
    console.warn('Invalid API URL format, using default:', error);
    return DEFAULT_API_URL;
  }
};

export const API_BASE_URL = getApiBaseUrl();

// Category options (matching backend data)
export const CATEGORIES = [
  { id: 0, name: 'Comedy' },
  { id: 1, name: 'Education' },
  { id: 2, name: 'Entertainment' },
  { id: 3, name: 'Gaming' },
  { id: 4, name: 'Howto & Style' },
  { id: 5, name: 'Music' },
  { id: 6, name: 'News & Politics' },
  { id: 7, name: 'Science & Technology' },
  { id: 8, name: 'Sports' },
  { id: 9, name: 'Travel & Events' },
];

