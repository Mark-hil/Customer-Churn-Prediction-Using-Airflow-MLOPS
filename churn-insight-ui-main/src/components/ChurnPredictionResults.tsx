import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, BarChart3, Target, Database } from 'lucide-react';

interface PredictionResult {
  churnProbability: number;
  riskLevel: 'low' | 'medium' | 'high';
  confidence: number;
  factors: {
    name: string;
    impact: number;
    positive: boolean;
  }[];
}

interface ChurnPredictionResultsProps {
  result: PredictionResult | null;
  isVisible: boolean;
  onRequestGroundTruth?: () => void;
}

const ChurnPredictionResults: React.FC<ChurnPredictionResultsProps> = ({ result, isVisible, onRequestGroundTruth }) => {
  if (!isVisible || !result) return null;

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'destructive';
      default: return 'secondary';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'low': return <CheckCircle className="h-5 w-5" />;
      case 'medium': return <AlertTriangle className="h-5 w-5" />;
      case 'high': return <TrendingUp className="h-5 w-5" />;
      default: return <BarChart3 className="h-5 w-5" />;
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

  return (
    <div className="space-y-6 animate-in slide-in-from-top duration-500">
      {/* Main Prediction Card */}
      <Card className="border-2 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-primary/10 to-accent/10 border-b">
          <CardTitle className="flex items-center gap-2 text-xl">
            <Target className="h-6 w-6 text-primary" />
            Churn Prediction Results
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Churn Probability */}
            <div className="text-center space-y-3">
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">Churn Probability</h3>
                <div className="text-3xl font-bold text-foreground">
                  {formatPercentage(result.churnProbability)}
                </div>
              </div>
              <Progress 
                value={result.churnProbability * 100} 
                className="h-3"
              />
            </div>

            {/* Risk Level */}
            <div className="text-center space-y-3">
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">Risk Level</h3>
                <div className="flex items-center justify-center gap-2">
                  <Badge variant={getRiskColor(result.riskLevel)} className="text-lg px-4 py-2 font-semibold">
                    {getRiskIcon(result.riskLevel)}
                    {result.riskLevel.toUpperCase()}
                  </Badge>
                </div>
              </div>
            </div>

            {/* Confidence */}
            <div className="text-center space-y-3">
              <div className="space-y-2">
                <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">Confidence</h3>
                <div className="text-3xl font-bold text-foreground">
                  {formatPercentage(result.confidence)}
                </div>
              </div>
              <Progress 
                value={result.confidence * 100} 
                className="h-3"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Level Alert */}
      <Alert className={`border-2 ${
        result.riskLevel === 'high' ? 'border-destructive/50 bg-destructive/10' :
        result.riskLevel === 'medium' ? 'border-warning/50 bg-warning/10' :
        'border-success/50 bg-success/10'
      }`}>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription className="font-medium">
          {result.riskLevel === 'high' && (
            <>
              <span className="text-destructive font-semibold">High Risk:</span> This customer has a high probability of churning. Immediate retention actions are recommended.
            </>
          )}
          {result.riskLevel === 'medium' && (
            <>
              <span className="text-warning font-semibold">Medium Risk:</span> This customer shows moderate churn risk. Consider proactive engagement strategies.
            </>
          )}
          {result.riskLevel === 'low' && (
            <>
              <span className="text-success font-semibold">Low Risk:</span> This customer has a low probability of churning. Continue with regular engagement.
            </>
          )}
        </AlertDescription>
      </Alert>

      {/* Key Factors */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            Key Factors Influencing Prediction
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {result.factors.map((factor, index) => (
            <div key={index} className="flex items-center justify-between p-3 rounded-lg border bg-card/50">
              <div className="flex items-center gap-3">
                {factor.positive ? (
                  <TrendingDown className="h-4 w-4 text-success" />
                ) : (
                  <TrendingUp className="h-4 w-4 text-destructive" />
                )}
                <span className="font-medium text-foreground">{factor.name}</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-24 bg-muted rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${factor.positive ? 'bg-success' : 'bg-destructive'}`}
                    style={{ width: `${Math.abs(factor.impact) * 100}%` }}
                  />
                </div>
                <Badge variant={factor.positive ? 'default' : 'destructive'} className="min-w-[60px] justify-center">
                  {factor.positive ? '-' : '+'}{Math.abs(factor.impact * 100).toFixed(1)}%
                </Badge>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Ground Truth Request */}
      {onRequestGroundTruth && (
        <Card className="border-warning/50 bg-warning/5">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Database className="h-5 w-5 text-warning" />
                <div>
                  <p className="font-medium text-foreground">Help improve our model</p>
                  <p className="text-sm text-muted-foreground">Record the actual outcome for this customer</p>
                </div>
              </div>
              <Button onClick={onRequestGroundTruth} variant="outline" className="border-warning text-warning hover:bg-warning/10">
                <Database className="h-4 w-4 mr-2" />
                Record Ground Truth
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ChurnPredictionResults;