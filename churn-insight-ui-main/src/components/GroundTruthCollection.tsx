import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { CheckCircle, XCircle, Database, Plus, Copy, Save, X, Edit2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

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
}

interface GroundTruthCollectionProps {
  customerData?: CustomerData;
  prediction?: PredictionResult;
  onSubmit?: () => void;
  onGroundTruthSubmit?: (groundTruth: boolean) => void;
}

interface GroundTruthData {
  customerId: string;
  actualChurn: boolean;
  dateRecorded: string;
  notes?: string;
  predictedChurn?: number;
  predictedRisk?: string;
}

const GroundTruthCollection: React.FC<GroundTruthCollectionProps> = ({ 
  customerData, 
  prediction, 
  onSubmit,
  onGroundTruthSubmit 
}) => {
  // Track if we're editing an existing record
  const [isEditing, setIsEditing] = useState(false);
  const [editingRecordId, setEditingRecordId] = useState<string | null>(null);
  
  // Auto-generate a customer ID if not provided
  const [customerId, setCustomerId] = useState(() => {
    // If we have customer data, generate a consistent ID
    if (customerData) {
      // Create a hash of customer data to generate a consistent ID
      const dataString = JSON.stringify(customerData);
      const hash = Array.from(dataString).reduce(
        (hash, char) => (hash << 5) - hash + char.charCodeAt(0),
        0
      );
      return `cust_${Math.abs(hash).toString(36).substring(0, 8)}`;
    }
    return `cust_${Date.now().toString(36).slice(-8)}`;
  });
  
  // Prevent modification of the customer ID
  const handleCustomerIdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Do nothing - prevent changes
    return;
  };
  const [actualChurn, setActualChurn] = useState<string>('');
  const [notes, setNotes] = useState('');
  const [groundTruthRecords, setGroundTruthRecords] = useState<GroundTruthData[]>([]);
  const { toast } = useToast();

  // Handle editing an existing record
  const handleEditRecord = (record: GroundTruthData) => {
    setCustomerId(record.customerId);
    setActualChurn(record.actualChurn ? 'true' : 'false');
    setNotes(record.notes || '');
    setEditingRecordId(record.customerId);
    setIsEditing(true);
    
    // Scroll to the form
    document.getElementById('ground-truth-form')?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle form submission for both new and edit
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!customerId || !actualChurn) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields.",
        variant: "destructive",
      });
      return;
    }

    const actualChurnBool = actualChurn === 'true';
    const now = new Date().toISOString().split('T')[0];
    
    const record: GroundTruthData = {
      customerId,
      actualChurn: actualChurnBool,
      dateRecorded: now,
      notes: notes || undefined,
      predictedChurn: prediction?.churnProbability,
      predictedRisk: prediction?.riskLevel,
    };

    if (isEditing && editingRecordId) {
      // Update existing record
      setGroundTruthRecords(prev => 
        prev.map(r => r.customerId === editingRecordId ? { ...record } : r)
      );
      
      toast({
        title: "Ground Truth Updated",
        description: `Updated record for customer ${customerId}`,
      });
    } else {
      // Add new record
      setGroundTruthRecords(prev => [record, ...prev]);
      
      toast({
        title: "Ground Truth Recorded",
        description: `Customer ${customerId} outcome has been saved.`,
      });
    }
    
    // Reset form
    if (!customerData) {
      // Only reset customer ID if we're not in the context of a specific customer
      setCustomerId(`cust_${Date.now().toString(36).slice(-8)}`);
    }
    setActualChurn('');
    setNotes('');
    setIsEditing(false);
    setEditingRecordId(null);

    // Call parent callbacks if provided
    onSubmit?.();
    
    // Call the ground truth submission handler if provided
    if (onGroundTruthSubmit) {
      onGroundTruthSubmit(actualChurnBool, isEditing);
    }
  };

  return (
    <div className="space-y-6">
      {/* Collection Form */}
      <Card className="bg-gradient-to-br from-card to-card/80 border-primary/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            Ground Truth Collection
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Record actual customer churn outcomes to validate model performance
          </p>
        </CardHeader>
        <CardContent>
          <form id="ground-truth-form" onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="customerId">Customer ID *</Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="customerId"
                    value={customerId}
                    onChange={handleCustomerIdChange}
                    placeholder="Auto-generated customer ID"
                    className="bg-muted/50 cursor-not-allowed"
                    readOnly
                    required
                  />
                  <Button 
                    type="button" 
                    variant="outline" 
                    size="sm"
                    onClick={() => navigator.clipboard.writeText(customerId)}
                    title="Copy to clipboard"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="actualChurn">Actual Outcome *</Label>
                <Select value={actualChurn} onValueChange={setActualChurn} required>
                  <SelectTrigger>
                    <SelectValue placeholder="Select outcome" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="false">Customer Retained</SelectItem>
                    <SelectItem value="true">Customer Churned</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="notes">Additional Notes</Label>
              <Input
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Optional notes about the outcome"
              />
            </div>

            <div className="flex gap-2">
              <Button type="submit" className="flex-1">
                {isEditing ? (
                  <>
                    <Save className="h-4 w-4 mr-2" />
                    Update Record
                  </>
                ) : (
                  <>
                    <Plus className="h-4 w-4 mr-2" />
                    Record Ground Truth
                  </>
                )}
              </Button>
              {isEditing && (
                <Button 
                  type="button" 
                  variant="outline" 
                  onClick={() => {
                    setActualChurn('');
                    setNotes('');
                    setIsEditing(false);
                    setEditingRecordId(null);
                  }}
                >
                  <X className="h-4 w-4 mr-2" />
                  Cancel
                </Button>
              )}
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Records Display */}
      {groundTruthRecords.length > 0 && (
        <Card className="bg-gradient-to-br from-card to-card/80">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Recent Records</span>
              <Badge variant="secondary">{groundTruthRecords.length} total</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {groundTruthRecords.map((record, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg hover:bg-muted/50 transition-colors">
                  <div className="flex items-center gap-3">
                    <div className={`p-1 rounded-full ${record.actualChurn ? 'bg-destructive/10' : 'bg-success/10'}`}>
                      {record.actualChurn ? (
                        <XCircle className="h-4 w-4 text-destructive" />
                      ) : (
                        <CheckCircle className="h-4 w-4 text-success" />
                      )}
                    </div>
                    <div>
                      <p className="font-medium">Customer {record.customerId}</p>
                      <p className="text-sm text-muted-foreground">
                        {record.actualChurn ? 'Churned' : 'Retained'} • {record.dateRecorded}
                      </p>
                      {record.notes && (
                        <p className="text-xs text-muted-foreground mt-1">{record.notes}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={record.actualChurn ? "destructive" : "success"}>
                      {record.actualChurn ? 'Churned' : 'Retained'}
                    </Badge>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => handleEditRecord(record)}
                      title="Edit this record"
                      className="h-8 w-8 p-0"
                    >
                      <Edit2 className="h-3.5 w-3.5" />
                      <span className="sr-only">Edit</span>
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default GroundTruthCollection;