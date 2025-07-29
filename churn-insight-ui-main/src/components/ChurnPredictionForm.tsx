import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { UserCheck, DollarSign, Phone, Wifi, FileText } from 'lucide-react';

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

interface ChurnPredictionFormProps {
  onPredict: (data: CustomerData) => void;
  isLoading?: boolean;
}

const ChurnPredictionForm: React.FC<ChurnPredictionFormProps> = ({ onPredict, isLoading = false }) => {
  const [formData, setFormData] = useState<CustomerData>({
    SeniorCitizen: 0,
    tenure: 40,
    MonthlyCharges: 61.2,
    TotalCharges: 2448.0,
    gender: "Female",
    Partner: "Yes",
    Dependents: "Yes",
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "No",
    DeviceProtection: "Yes",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "No",
    Contract: "One year",
    PaperlessBilling: "No",
    PaymentMethod: "Bank transfer (automatic)"
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(formData);
  };

  const updateField = (field: keyof CustomerData, value: string | number) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <Card className="w-full">
      <CardHeader className="bg-gradient-to-r from-primary/10 to-accent/10 border-b">
        <CardTitle className="flex items-center gap-2 text-xl">
          <UserCheck className="h-6 w-6 text-primary" />
          Customer Information
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Demographics Section */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <UserCheck className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">Demographics</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label htmlFor="gender">Gender</Label>
                <Select value={formData.gender} onValueChange={(value) => updateField('gender', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Male">Male</SelectItem>
                    <SelectItem value="Female">Female</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="seniorCitizen">Senior Citizen</Label>
                <Select value={formData.SeniorCitizen.toString()} onValueChange={(value) => updateField('SeniorCitizen', parseInt(value))}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">No</SelectItem>
                    <SelectItem value="1">Yes</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="partner">Partner</Label>
                <Select value={formData.Partner} onValueChange={(value) => updateField('Partner', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="dependents">Dependents</Label>
                <Select value={formData.Dependents} onValueChange={(value) => updateField('Dependents', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <Separator />

          {/* Financial Information */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <DollarSign className="h-5 w-5 text-success" />
              <h3 className="text-lg font-semibold text-foreground">Financial Information</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="tenure">Tenure (months)</Label>
                <Input
                  id="tenure"
                  type="number"
                  value={formData.tenure}
                  onChange={(e) => updateField('tenure', parseInt(e.target.value) || 0)}
                  className="bg-background"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="monthlyCharges">Monthly Charges ($)</Label>
                <Input
                  id="monthlyCharges"
                  type="number"
                  step="0.01"
                  value={formData.MonthlyCharges}
                  onChange={(e) => updateField('MonthlyCharges', parseFloat(e.target.value) || 0)}
                  className="bg-background"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="totalCharges">Total Charges ($)</Label>
                <Input
                  id="totalCharges"
                  type="number"
                  step="0.01"
                  value={formData.TotalCharges}
                  onChange={(e) => updateField('TotalCharges', parseFloat(e.target.value) || 0)}
                  className="bg-background"
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Services Section */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <Phone className="h-5 w-5 text-accent" />
              <h3 className="text-lg font-semibold text-foreground">Phone Services</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="phoneService">Phone Service</Label>
                <Select value={formData.PhoneService} onValueChange={(value) => updateField('PhoneService', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="multipleLines">Multiple Lines</Label>
                <Select value={formData.MultipleLines} onValueChange={(value) => updateField('MultipleLines', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="No phone service">No phone service</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <Separator />

          {/* Internet Services */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <Wifi className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">Internet Services</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="internetService">Internet Service</Label>
                <Select value={formData.InternetService} onValueChange={(value) => updateField('InternetService', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="DSL">DSL</SelectItem>
                    <SelectItem value="Fiber optic">Fiber optic</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="onlineSecurity">Online Security</Label>
                <Select value={formData.OnlineSecurity} onValueChange={(value) => updateField('OnlineSecurity', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="No internet service">No internet service</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="onlineBackup">Online Backup</Label>
                <Select value={formData.OnlineBackup} onValueChange={(value) => updateField('OnlineBackup', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="No internet service">No internet service</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="deviceProtection">Device Protection</Label>
                <Select value={formData.DeviceProtection} onValueChange={(value) => updateField('DeviceProtection', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="No internet service">No internet service</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="techSupport">Tech Support</Label>
                <Select value={formData.TechSupport} onValueChange={(value) => updateField('TechSupport', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="No internet service">No internet service</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="streamingTV">Streaming TV</Label>
                <Select value={formData.StreamingTV} onValueChange={(value) => updateField('StreamingTV', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="No internet service">No internet service</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="streamingMovies">Streaming Movies</Label>
                <Select value={formData.StreamingMovies} onValueChange={(value) => updateField('StreamingMovies', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                    <SelectItem value="No internet service">No internet service</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <Separator />

          {/* Contract & Billing */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 mb-4">
              <FileText className="h-5 w-5 text-warning" />
              <h3 className="text-lg font-semibold text-foreground">Contract & Billing</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="contract">Contract</Label>
                <Select value={formData.Contract} onValueChange={(value) => updateField('Contract', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Month-to-month">Month-to-month</SelectItem>
                    <SelectItem value="One year">One year</SelectItem>
                    <SelectItem value="Two year">Two year</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="paperlessBilling">Paperless Billing</Label>
                <Select value={formData.PaperlessBilling} onValueChange={(value) => updateField('PaperlessBilling', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="paymentMethod">Payment Method</Label>
                <Select value={formData.PaymentMethod} onValueChange={(value) => updateField('PaymentMethod', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Electronic check">Electronic check</SelectItem>
                    <SelectItem value="Mailed check">Mailed check</SelectItem>
                    <SelectItem value="Bank transfer (automatic)">Bank transfer (automatic)</SelectItem>
                    <SelectItem value="Credit card (automatic)">Credit card (automatic)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <div className="pt-6">
            <Button 
              type="submit" 
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-3 h-12 text-lg"
              disabled={isLoading}
            >
              {isLoading ? 'Analyzing...' : 'Predict Churn Risk'}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};

export default ChurnPredictionForm;