import React, { useState, useEffect } from 'react';
import ChurnPredictionForm from './ChurnPredictionForm';
import ChurnPredictionResults from './ChurnPredictionResults';
import GroundTruthCollection from './GroundTruthCollection';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BarChart3, Users, TrendingUp, AlertTriangle, Brain, Zap, AlertCircle } from 'lucide-react';
import { useApi } from '@/hooks/useApi';
import { toast } from 'sonner';

interface CustomerData {
  SeniorCitizen: number;
  tenure: number;
  MonthlyCharges: number;
  TotalCharges: number;
  gender: string;
  Partner: string;
  Dependents: string;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
}

interface PredictionResult {
  churnProbability: number;
  riskLevel: 'low' | 'medium' | 'high';
  confidence: number;
  factors: {
    name: string;
    impact: number;
    positive: boolean;
  }[];
  requestId?: string;
  modelType?: string;
  modelName?: string;
}

const ChurnDashboard: React.FC = () => {
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [currentCustomerData, setCurrentCustomerData] = useState<CustomerData | null>(null);
  const [showGroundTruthForm, setShowGroundTruthForm] = useState(false);
  const [showResults, setShowResults] = useState(false);
  
  const { 
    makePrediction, 
    submitGroundTruth, 
    checkHealth, 
    isHealthy, 
    isLoading, 
    error, 
    resetError 
  } = useApi();

  // Check API health on component mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const isApiHealthy = await checkHealth();
        if (!isApiHealthy) {
          toast.error('API is not available. Some features may be limited.');
        }
      } catch (err) {
        console.error('API health check failed:', err);
        toast.error('Failed to connect to the prediction service');
      }
    };

    checkApiHealth();
  }, [checkHealth]);

  // Handle API errors
  useEffect(() => {
    if (error) {
      toast.error(error);
      resetError();
    }
  }, [error, resetError]);

  // Mock prediction function - replace with actual API call
  const mockPredict = (data: CustomerData): PredictionResult => {
    // Simulate complex prediction logic
    let churnScore = 0;
    
    // Contract type impact
    if (data.Contract === 'Month-to-month') churnScore += 0.3;
    else if (data.Contract === 'One year') churnScore += 0.1;
    
    // Tenure impact
    if (data.tenure < 12) churnScore += 0.25;
    else if (data.tenure < 24) churnScore += 0.1;
    else churnScore -= 0.15;
    
    // Services impact
    if (data.OnlineSecurity === 'No') churnScore += 0.1;
    if (data.TechSupport === 'No') churnScore += 0.1;
    if (data.OnlineBackup === 'No') churnScore += 0.05;
    
    // Demographics
    if (data.SeniorCitizen === 1) churnScore += 0.1;
    if (data.Partner === 'No') churnScore += 0.05;
    if (data.Dependents === 'No') churnScore += 0.05;
    
    // Billing
    if (data.PaperlessBilling === 'Yes') churnScore += 0.05;
    if (data.PaymentMethod === 'Electronic check') churnScore += 0.1;
    
    // Normalize score
    churnScore = Math.max(0, Math.min(1, churnScore));
    
    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high' = 'low';
    if (churnScore > 0.7) riskLevel = 'high';
    else if (churnScore > 0.4) riskLevel = 'medium';
    
    // Generate factors
    const factors = [
      { name: 'Contract Type', impact: data.Contract === 'Month-to-month' ? 0.3 : -0.2, positive: data.Contract !== 'Month-to-month' },
      { name: 'Customer Tenure', impact: data.tenure > 24 ? -0.25 : 0.2, positive: data.tenure > 24 },
      { name: 'Online Security', impact: data.OnlineSecurity === 'Yes' ? -0.15 : 0.15, positive: data.OnlineSecurity === 'Yes' },
      { name: 'Tech Support', impact: data.TechSupport === 'Yes' ? -0.1 : 0.1, positive: data.TechSupport === 'Yes' },
      { name: 'Payment Method', impact: data.PaymentMethod === 'Electronic check' ? 0.12 : -0.08, positive: data.PaymentMethod !== 'Electronic check' },
    ].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

    return {
      churnProbability: churnScore,
      riskLevel,
      confidence: 0.85 + Math.random() * 0.1, // Simulate confidence between 85-95%
      factors
    };
  };

  const handlePredict = async (data: CustomerData) => {
    try {
      setShowResults(false);
      setShowGroundTruthForm(false);
      setCurrentCustomerData(data);
      
      // Convert form data to match API expected format
      const apiInput = {
        ...data,
        // Convert string 'Yes'/'No' to 1/0 if needed by your API
        SeniorCitizen: data.SeniorCitizen,
        tenure: data.tenure,
        MonthlyCharges: data.MonthlyCharges,
        TotalCharges: data.TotalCharges,
      };

      // Make the API call
      const response = await makePrediction(apiInput);
      console.log('Full prediction response:', JSON.stringify(response, null, 2));
      
      if (response && response.predictions && response.predictions.length > 0) {
        const prediction = response.predictions[0];
        
        // Get the request ID from the response or generate a new one
        const requestId = response._requestId || 
                         (response._headers && response._headers['x-request-id']) || 
                         (response.headers && response.headers.get('x-request-id')) ||
                         crypto.randomUUID();
        
        console.log('Using request ID for ground truth:', {
          requestId,
          hasRequestId: !!requestId,
          fromHeaders: response._headers?.['x-request-id'],
          responseKeys: Object.keys(response)
        });
        
        if (!requestId) {
          console.warn('No request ID found in response, using generated ID');
        }
        
        // Map API response to our frontend model
        const result: PredictionResult = {
          churnProbability: prediction.probability,
          riskLevel: prediction.probability > 0.7 ? 'high' as const : prediction.probability > 0.3 ? 'medium' as const : 'low' as const,
          confidence: prediction.probability,
          factors: [
            { name: 'Contract Type', impact: prediction.probability, positive: prediction.probability > 0.5 },
            { name: 'Customer Tenure', impact: prediction.probability, positive: prediction.probability > 0.5 },
            { name: 'Monthly Charges', impact: prediction.probability, positive: prediction.probability > 0.5 },
            { name: 'Payment Method', impact: prediction.probability, positive: prediction.probability > 0.5 },
            { name: 'Services', impact: prediction.probability, positive: prediction.probability > 0.5 },
          ],
          requestId: requestId,
          modelType: prediction.model_type,
          modelName: prediction.model_name,
        };
        
        setPredictionResult(result);
        setShowResults(true);
      }
    } catch (err) {
      console.error('Prediction failed:', err);
      toast.error('Failed to get prediction. Please try again.');
    }
  };

  const resetPrediction = () => {
    setPredictionResult(null);
    setShowResults(false);
    setCurrentCustomerData(null);
    setShowGroundTruthForm(false);
  };

  const handleRequestGroundTruth = () => {
    setShowGroundTruthForm(true);
  };

  const handleGroundTruthSubmitted = async (groundTruth: boolean, isUpdate: boolean = false) => {
    if (!predictionResult) {
      toast.error('No prediction data available to submit feedback');
      return;
    }

    console.log(`${isUpdate ? 'Updating' : 'Submitting'} ground truth with prediction result:`, {
      requestId: predictionResult.requestId,
      modelName: predictionResult.modelName,
      modelType: predictionResult.modelType,
      groundTruth: groundTruth ? 1 : 0,
      isUpdate
    });

    if (!predictionResult.requestId) {
      toast.error('Missing request ID. Cannot submit feedback.');
      return;
    }

    try {
      // Convert boolean to number (1 for true, 0 for false)
      const groundTruthValue = groundTruth ? 1 : 0;
      
      console.log(`Calling ${isUpdate ? 'update' : 'submit'}GroundTruth with:`, {
        requestId: predictionResult.requestId,
        groundTruth: groundTruthValue
      });
      
      // For now, we'll use the same endpoint for both create and update
      // The backend should handle updates to existing ground truth records
      await submitGroundTruth(
        predictionResult.requestId, 
        groundTruthValue
      );
      
      toast.success(
        isUpdate 
          ? 'Ground truth updated successfully!' 
          : 'Thank you for your feedback! It will help improve our model.'
      );
      
      if (!isUpdate) {
        setShowGroundTruthForm(false);
      }
      
      console.log(`Ground truth ${isUpdate ? 'updated' : 'submitted'} successfully for request ID:`, predictionResult.requestId);
      
      // Update the prediction result with the latest ground truth
      setPredictionResult(prev => ({
        ...prev!,
        groundTruthSubmitted: true,
        groundTruthValue: groundTruthValue,
        lastUpdated: new Date().toISOString()
      }));
    } catch (err) {
      console.error('Failed to submit ground truth:', err);
      toast.error('Failed to submit feedback. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-muted/30 to-background">
      {/* Header */}
      <div className="bg-card/50 backdrop-blur-sm border-b sticky top-0 z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-primary to-accent rounded-lg">
                <Brain className="h-8 w-8 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                  Churn Prediction Analytics
                </h1>
                <p className="text-muted-foreground mt-1">AI-powered customer retention insights</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="secondary" className="px-3 py-1">
                <Zap className="h-4 w-4 mr-1" />
                ML Model v2.1
              </Badge>
              {showResults && (
                <Button variant="outline" onClick={resetPrediction}>
                  New Prediction
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-gradient-to-br from-card to-card/80 border-primary/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Users className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Total Customers</p>
                  <p className="text-2xl font-bold text-foreground">7,043</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-card to-card/80 border-success/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-success/10 rounded-lg">
                  <TrendingUp className="h-5 w-5 text-success" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Retention Rate</p>
                  <p className="text-2xl font-bold text-foreground">73.4%</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-card to-card/80 border-warning/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-warning/10 rounded-lg">
                  <AlertTriangle className="h-5 w-5 text-warning" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">At Risk</p>
                  <p className="text-2xl font-bold text-foreground">1,869</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-card to-card/80 border-accent/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-accent/10 rounded-lg">
                  <BarChart3 className="h-5 w-5 text-accent" />
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg. Accuracy</p>
                  <p className="text-2xl font-bold text-foreground">94.2%</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content with Tabs */}
        <Tabs defaultValue="prediction" className="space-y-6">
          <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto">
            <TabsTrigger value="prediction">Churn Prediction</TabsTrigger>
            <TabsTrigger value="groundtruth">Ground Truth</TabsTrigger>
          </TabsList>
          
          <TabsContent value="prediction" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Form Section */}
              <div className="space-y-6">
                <ChurnPredictionForm onPredict={handlePredict} isLoading={isLoading} />
              </div>

              {/* Results Section */}
              <div className="space-y-6">
                {!showResults && !isLoading && (
                  <Card className="h-full flex items-center justify-center border-2 border-dashed border-muted-foreground/25">
                    <CardContent className="text-center p-8">
                      <Brain className="h-16 w-16 text-muted-foreground/50 mx-auto mb-4" />
                      <h3 className="text-lg font-semibold text-muted-foreground mb-2">Ready for Analysis</h3>
                      <p className="text-muted-foreground">Enter customer data and click "Predict Churn Risk" to see AI-powered insights</p>
                    </CardContent>
                  </Card>
                )}

                {isLoading && (
                  <Card className="h-full flex items-center justify-center">
                    <CardContent className="text-center p-8">
                      <div className="animate-spin rounded-full h-16 w-16 border-4 border-primary border-t-transparent mx-auto mb-4"></div>
                      <h3 className="text-lg font-semibold text-foreground mb-2">Analyzing Customer Data</h3>
                      <p className="text-muted-foreground">Our AI model is processing the information...</p>
                    </CardContent>
                  </Card>
                )}

                <ChurnPredictionResults 
                  result={predictionResult} 
                  isVisible={showResults} 
                  onRequestGroundTruth={handleRequestGroundTruth}
                />
                
                {showGroundTruthForm && currentCustomerData && predictionResult && (
                  <Card className="mt-6">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <AlertTriangle className="h-5 w-5 text-warning" />
                        Ground Truth Collection
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <GroundTruthCollection 
                        customerData={currentCustomerData}
                        prediction={{
                          ...predictionResult,
                          // Ensure we have the required properties
                          churnProbability: predictionResult.churnProbability || 0,
                          riskLevel: predictionResult.riskLevel || 'medium',
                          confidence: predictionResult.confidence || 0.5,
                          factors: predictionResult.factors || []
                        }}
                        onSubmit={() => {
                          // This will be called when the form is submitted in the GroundTruthCollection component
                          // The actual ground truth value is handled within the component
                          console.log('Ground truth form submitted');
                        }}
                        onGroundTruthSubmit={handleGroundTruthSubmitted} // New prop for ground truth submission
                      />
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="groundtruth" className="space-y-6">
            <div className="max-w-4xl mx-auto">
              <GroundTruthCollection 
                onGroundTruthSubmit={handleGroundTruthSubmitted}
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default ChurnDashboard;