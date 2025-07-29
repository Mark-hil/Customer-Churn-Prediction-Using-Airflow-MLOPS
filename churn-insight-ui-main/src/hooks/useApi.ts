import { useState, useCallback } from 'react';
import { predict, updateGroundTruth, healthCheck } from '@/services/api';
import type { PredictionInput, PredictionResponse } from '@/services/api';

export const useApi = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  const makePrediction = useCallback(async (inputData: Record<string, any>, shadowMode = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await predict(inputData, shadowMode);
      setPrediction(result);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const submitGroundTruth = useCallback(async (requestId: string, groundTruth: number) => {
    setIsLoading(true);
    setError(null);
    try {
      await updateGroundTruth(requestId, groundTruth);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to submit feedback';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const checkHealth = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      await healthCheck();
      setIsHealthy(true);
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Health check failed';
      setError(errorMessage);
      setIsHealthy(false);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    makePrediction,
    submitGroundTruth,
    checkHealth,
    prediction,
    isHealthy,
    isLoading,
    error,
    resetError: () => setError(null),
  };
};

export default useApi;
