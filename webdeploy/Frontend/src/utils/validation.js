// Input validation and sanitization utilities

/**
 * Validate and sanitize title input
 * @param {string} title - Video title
 * @returns {object} - { valid: boolean, sanitized: string, error: string }
 */
export const validateTitle = (title) => {
  const maxLength = 200;
  const trimmed = title.trim();
  
  if (!trimmed) {
    return { valid: false, sanitized: '', error: 'Title is required' };
  }
  
  if (trimmed.length > maxLength) {
    return { valid: false, sanitized: trimmed.substring(0, maxLength), error: `Title must be ${maxLength} characters or less` };
  }
  
  // Sanitize: remove potentially dangerous characters but allow normal text
  const sanitized = trimmed.replace(/[<>]/g, '');
  
  return { valid: true, sanitized, error: null };
};

/**
 * Validate and sanitize hashtags input
 * @param {string} hashtags - Hashtags string
 * @returns {object} - { valid: boolean, sanitized: string, error: string }
 */
export const validateHashtags = (hashtags) => {
  const maxLength = 500;
  const trimmed = hashtags.trim();
  
  if (trimmed.length > maxLength) {
    return { valid: false, sanitized: trimmed.substring(0, maxLength), error: `Hashtags must be ${maxLength} characters or less` };
  }
  
  // Sanitize: remove potentially dangerous characters
  const sanitized = trimmed.replace(/[<>]/g, '');
  
  return { valid: true, sanitized, error: null };
};

/**
 * Validate duration in seconds
 * @param {number} duration - Duration in seconds
 * @returns {object} - { valid: boolean, sanitized: number, error: string }
 */
export const validateDuration = (duration) => {
  const minDuration = 0;
  const maxDuration = 86400; // 24 hours in seconds
  
  const num = Number(duration);
  
  if (isNaN(num)) {
    return { valid: false, sanitized: 0, error: 'Duration must be a number' };
  }
  
  if (num < minDuration) {
    return { valid: false, sanitized: minDuration, error: `Duration must be at least ${minDuration} seconds` };
  }
  
  if (num > maxDuration) {
    return { valid: false, sanitized: maxDuration, error: `Duration must be at most ${maxDuration} seconds (24 hours)` };
  }
  
  return { valid: true, sanitized: Math.floor(num), error: null };
};

/**
 * Sanitize error message for display to users
 * @param {string} error - Error message from API or system
 * @returns {string} - User-friendly error message
 */
export const sanitizeErrorMessage = (error) => {
  if (!error) {
    return 'An unexpected error occurred. Please try again.';
  }
  
  // Convert error to string
  const errorStr = typeof error === 'string' ? error : error.toString();
  
  // Map common backend errors to user-friendly messages
  const errorMappings = {
    'network': 'Network error. Please check your connection.',
    'timeout': 'Request timed out. Please try again.',
    '500': 'Server error. Please try again later.',
    '503': 'Service temporarily unavailable. Please try again later.',
    '404': 'Resource not found.',
    '400': 'Invalid request. Please check your input.',
    '401': 'Authentication failed.',
    '403': 'Access denied.',
  };
  
  // Check for known error patterns
  for (const [key, message] of Object.entries(errorMappings)) {
    if (errorStr.toLowerCase().includes(key.toLowerCase())) {
      return message;
    }
  }
  
  // Remove potentially sensitive information (paths, stack traces, etc.)
  let sanitized = errorStr
    .replace(/\/[\w\/\-\.]+/g, '[path]') // Replace file paths
    .replace(/at\s+[\w\.]+/g, '') // Remove stack trace indicators
    .replace(/Error:\s*/gi, '') // Remove error: prefix
    .trim();
  
  // Limit error message length
  if (sanitized.length > 200) {
    sanitized = sanitized.substring(0, 200) + '...';
  }
  
  // If error still looks technical, return generic message
  if (sanitized.includes('[path]') || sanitized.includes('Traceback') || sanitized.includes('File')) {
    return 'An unexpected error occurred. Please try again.';
  }
  
  return sanitized || 'An unexpected error occurred. Please try again.';
};

/**
 * Check if we're in production environment
 * @returns {boolean}
 */
export const isProduction = () => {
  return process.env.NODE_ENV === 'production';
};

