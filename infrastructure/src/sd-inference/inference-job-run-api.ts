import { PythonFunction, PythonFunctionProps } from '@aws-cdk/aws-lambda-python-alpha';
import {
  Aws,
  aws_apigateway as apigw,
  aws_apigateway,
  aws_dynamodb,
  aws_iam,
  aws_lambda,
  aws_s3,
  Duration,
} from 'aws-cdk-lib';
import { MethodOptions } from 'aws-cdk-lib/aws-apigateway/lib/method';
import { Effect } from 'aws-cdk-lib/aws-iam';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Architecture, Runtime } from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';

export interface RunInferenceJobApiProps{
  router: aws_apigateway.Resource;
  httpMethod: string;
  endpointDeploymentTable: aws_dynamodb.Table;
  inferenceJobTable: aws_dynamodb.Table;
  checkpointTable: aws_dynamodb.Table;
  srcRoot: string;
  s3Bucket: aws_s3.Bucket;
  commonLayer: aws_lambda.LayerVersion;

}

export class RunInferenceJobApi {

  private readonly id: string;
  private readonly scope: Construct;
  private readonly srcRoot: string;
  private readonly layer: aws_lambda.LayerVersion;
  private readonly s3Bucket: aws_s3.Bucket;
  private readonly httpMethod: string;
  private readonly router: aws_apigateway.Resource;
  private readonly endpointDeploymentTable: aws_dynamodb.Table;
  private readonly inferenceJobTable: aws_dynamodb.Table;
  private readonly checkpointTable: aws_dynamodb.Table;

  constructor(scope: Construct, id: string, props: RunInferenceJobApiProps) {
    this.id = id;
    this.scope = scope;
    this.srcRoot = props.srcRoot;
    this.endpointDeploymentTable = props.endpointDeploymentTable;
    this.router = props.router;
    this.inferenceJobTable = props.inferenceJobTable;
    this.checkpointTable = props.checkpointTable;
    this.layer = props.commonLayer;
    this.s3Bucket = props.s3Bucket;
    this.httpMethod = props.httpMethod;

    this.updateTrainJobLambda();
  }


  private getLambdaRole(): aws_iam.Role {
    const newRole = new aws_iam.Role(this.scope, `${this.id}-role`, {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
    newRole.addToPolicy(new aws_iam.PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        'dynamodb:BatchGetItem',
        'dynamodb:GetItem',
        'dynamodb:Scan',
        'dynamodb:Query',
        'dynamodb:BatchWriteItem',
        'dynamodb:PutItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem',
      ],
      resources: [
        this.inferenceJobTable.tableArn,
        this.endpointDeploymentTable.tableArn,
        this.checkpointTable.tableArn,
      ],
    }));

    newRole.addToPolicy(new aws_iam.PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        'sagemaker:InvokeEndpointAsync',
      ],
      resources: [`arn:aws:sagemaker:${Aws.REGION}:${Aws.ACCOUNT_ID}:endpoint/*`],
    }));

    newRole.addToPolicy(new aws_iam.PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        's3:GetObject',
        's3:PutObject',
        's3:DeleteObject',
        's3:ListBucket',
        's3:CreateBucket',
      ],
      resources: [
        `${this.s3Bucket.bucketArn}/*`,
        'arn:aws:s3:::*SageMaker*',
        'arn:aws:s3:::*Sagemaker*',
        'arn:aws:s3:::*sagemaker*',
      ],
    }));

    newRole.addToPolicy(new aws_iam.PolicyStatement({
      effect: Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'kms:Decrypt',
      ],
      resources: ['*'],
    }));

    return newRole;
  }

  private updateTrainJobLambda(): aws_lambda.IFunction {
    const lambdaFunction = new PythonFunction(this.scope, `${this.id}-updateTrainJob`, <PythonFunctionProps>{
      functionName: `${this.id}-run-infer-job-v2`,
      entry: `${this.srcRoot}/inference_v2`,
      architecture: Architecture.X86_64,
      runtime: Runtime.PYTHON_3_9,
      index: 'inference_api.py',
      handler: 'run_inference',
      timeout: Duration.seconds(900),
      role: this.getLambdaRole(),
      memorySize: 1024,
      environment: {
        S3_BUCKET: this.s3Bucket.bucketName,
        DDB_ENDPOINT_DEPLOYMENT_TABLE_NAME: this.endpointDeploymentTable.tableName,
        DDB_INFERENCE_TABLE_NAME: this.inferenceJobTable.tableName,
        CHECKPOINT_TABLE: this.checkpointTable.tableName,
      },
      layers: [this.layer],
    });

    const runJobIntegration = new apigw.LambdaIntegration(
      lambdaFunction,
      {
        proxy: false,
        requestParameters: {
          'integration.request.path.inference_id': 'method.request.path.inference_id',
        },
        requestTemplates: {
          'application/json': '{\n' +
              '    "pathStringParameters": {\n' +
              '        #foreach($pathParam in $input.params().path.keySet())\n' +
              '        "$pathParam": "$util.escapeJavaScript($input.params().path.get($pathParam))"\n' +
              '        #if($foreach.hasNext),#end\n' +
              '        #end\n' +
              '    }\n' +
              '}',
        },
        integrationResponses: [{ statusCode: '200' }],
      },
    );

    const inferIdRouter = this.router.addResource('{inference_id}');
    inferIdRouter.addResource('run').addMethod(this.httpMethod, runJobIntegration, <MethodOptions>{
      apiKeyRequired: true,
      requestParameters: {
        'method.request.path.inference_id': true,
      },
      methodResponses: [{
        statusCode: '200',
      }, { statusCode: '500' }],
    });
    return lambdaFunction;
  }
}
