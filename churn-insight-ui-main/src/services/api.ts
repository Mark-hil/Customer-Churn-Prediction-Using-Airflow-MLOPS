const API_BASE_URL = 'http://localhost:8083/api/v1';

// Generate a UUID using browser's crypto API
const generateUUID = () => {
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);
  
  // Convert bytes to hex string
  let hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
  
  // Insert hyphens in the correct positions
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20)
  ].join('-');
};

export interface PredictionInput {
  // Define the input fields based on your model's requirements
  // Example:
  // age: number;
  // gender: string;
  // account_balance: number;
  [key: string]: any;
}

export interface PredictionResult {
  model_type: string;
  prediction: number;
  probability: number;
  prediction_time: string;
  model_name: string;
  model_version: string;
  interpretation: string;
  ab_test_group?: string;
}

export interface PredictionResponse {
  predictions: PredictionResult[];
  production_model: string;
  shadow_mode: boolean;
}

export const predict = async (inputData: PredictionInput, shadowMode: boolean = false): Promise<PredictionResponse> => {
  const requestId = generateUUID();
  const url = `${API_BASE_URL}/predict`;
  const requestBody = {
    input_data: inputData,
    shadow_mode: shadowMode,
  };

  console.log('Making prediction request:', { url, requestId, requestBody });

  try {
    // Get A/B test group from environment variable, default to 'B' if not set
    const abTestGroup = import.meta.env.VITE_AB_TEST_GROUP;
    
    console.log('Using A/B test group:', abTestGroup);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId,
        'X-AB-Test': abTestGroup,
      },
      body: JSON.stringify(requestBody),
    });

    const responseText = await response.text();
    let result;
    
    try {
      result = responseText ? JSON.parse(responseText) : {};
    } catch (e) {
      console.error('Failed to parse response as JSON:', responseText);
      throw new Error(`Invalid response format: ${responseText.substring(0, 200)}`);
    }

    if (!response.ok) {
      console.error('Prediction API error:', {
        status: response.status,
        statusText: response.statusText,
        response: result,
      });
      throw new Error(result.detail || `Prediction failed with status ${response.status}: ${response.statusText}`);
    }

    // Get the request ID from the response headers
    const responseHeaders: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });

    // Get the request ID from headers (case-insensitive)
    const getHeader = (name: string) => {
      const lowerName = name.toLowerCase();
      return Object.entries(responseHeaders).find(
        ([key]) => key.toLowerCase() === lowerName
      )?.[1];
    };

    const responseRequestId = getHeader('x-request-id');
    const finalRequestId = responseRequestId || requestId;

    console.log('Prediction API success:', { 
      ...result, 
      _headers: responseHeaders,
      _requestId: finalRequestId,
      originalRequestId: requestId,
      responseRequestId
    });
    
    // Return both the result and the headers
    return {
      ...result,
      _headers: responseHeaders,
      _requestId: finalRequestId
    };
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
};

export const updateGroundTruth = async (requestId: string, groundTruth: number): Promise<void> => {
  // Create URL with query parameters
  const params = new URLSearchParams({
    request_id: requestId,
    ground_truth: groundTruth.toString(),
  });
  
  const url = `${API_BASE_URL}/predict/update-ground-truth?${params.toString()}`;
  
  console.log('Submitting ground truth to:', url);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': crypto.randomUUID(), // Add a new request ID for this request
      },
      // No body needed as we're using query parameters
    });

    let responseData;
    try {
      responseData = await response.json();
    } catch (e) {
      console.error('Failed to parse response as JSON');
      throw new Error(`Invalid response format: ${await response.text()}`);
    }
    
    if (!response.ok) {
      console.error('Ground truth submission failed:', {
        status: response.status,
        statusText: response.statusText,
        response: responseData,
      });
      throw new Error(
        responseData.detail || 
        responseData.message || 
        `Failed to submit ground truth: ${response.status} ${response.statusText}`
      );
    }
    
    console.log('Ground truth updated successfully:', responseData);
  } catch (error) {
    console.error('Error submitting ground truth:', error);
    throw error;
  }
};

export const healthCheck = async (): Promise<{ status: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/predict/health`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Health check failed');
    }
    return response.json();
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
};
