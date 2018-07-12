// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <AssertMacros.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import <CoreImage/CoreImage.h>
#import <ImageIO/ImageIO.h>
#import "CameraViewController.h"

#include <sys/time.h>

#include "tensorflow_utils.h"

// If you have your own model, modify this to the file name, and make sure
// you've added the file to your app resources too.
static NSString* model_file_name = @"stripped_graph";
static NSString* model_file_type = @"pb";
// This controls whether we'll be loading a plain GraphDef proto, or a
// file created by the convert_graphdef_memmapped_format utility that wraps a
// GraphDef and parameter file that can be mapped into memory from file to
// reduce overall memory usage.
const bool model_uses_memory_mapping = false;
// If you have your own model, point this to the labels file.
static NSString* labels_file_name = @"imagenet_comp_graph_label_strings";
static NSString* labels_file_type = @"txt";
// These dimensions need to match those the model was trained with.
const int wanted_input_width = 299;
const int wanted_input_height = 299;
const int wanted_input_channels = 3;

// GL equivalent to substract 0.5 and then mult by 2
const float input_mean = 128;
const float input_std = 128;
const std::string input_layer_name = "Reshape";
const std::string output_layer_name = "inception_v3/logits/logits/xw_plus_b" ;

bool pauseu=false;  // should really replace with  if ([session isRunning]) {

static void *AVCaptureStillImageIsCapturingStillImageContext =
    &AVCaptureStillImageIsCapturingStillImageContext;

@interface CameraExampleViewController (InternalMethods) <UINavigationControllerDelegate,UIImagePickerControllerDelegate>
- (void)setupAVCapture;
- (void)teardownAVCapture;
@end

@implementation CameraExampleViewController


- (void)setupAVCapture {
  NSError *error = nil;

  session = [AVCaptureSession new];
  if ([[UIDevice currentDevice] userInterfaceIdiom] ==
      UIUserInterfaceIdiomPhone)
    [session setSessionPreset:AVCaptureSessionPreset640x480];
  else
    [session setSessionPreset:AVCaptureSessionPresetPhoto];

  AVCaptureDevice *device =
      [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
  AVCaptureDeviceInput *deviceInput =
      [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
  assert(error == nil);

  isUsingFrontFacingCamera = NO;
  if ([session canAddInput:deviceInput]) [session addInput:deviceInput];


  stillImageOutput = [AVCaptureStillImageOutput new];
  [stillImageOutput
      addObserver:self
       forKeyPath:@"capturingStillImage"
          options:NSKeyValueObservingOptionNew
          context:(void *)(AVCaptureStillImageIsCapturingStillImageContext)];
  if ([session canAddOutput:stillImageOutput])
    [session addOutput:stillImageOutput];

  videoDataOutput = [AVCaptureVideoDataOutput new];

  NSDictionary *rgbOutputSettings = [NSDictionary
      dictionaryWithObject:[NSNumber numberWithInt:kCMPixelFormat_32BGRA]
                    forKey:(id)kCVPixelBufferPixelFormatTypeKey];
  [videoDataOutput setVideoSettings:rgbOutputSettings];
  [videoDataOutput setAlwaysDiscardsLateVideoFrames:YES];
  videoDataOutputQueue =
      dispatch_queue_create("VideoDataOutputQueue", DISPATCH_QUEUE_SERIAL);
  [videoDataOutput setSampleBufferDelegate:self queue:videoDataOutputQueue];

  if ([session canAddOutput:videoDataOutput])
    [session addOutput:videoDataOutput];
  [[videoDataOutput connectionWithMediaType:AVMediaTypeVideo] setEnabled:YES];

  previewLayer = [[AVCaptureVideoPreviewLayer alloc] initWithSession:session];
  [previewLayer setBackgroundColor:[[UIColor blackColor] CGColor]];
  [previewLayer setVideoGravity:AVLayerVideoGravityResizeAspect];
  CALayer *rootLayer = [previewView layer];
  [rootLayer setMasksToBounds:YES];
  [previewLayer setFrame:[rootLayer bounds]];
  [rootLayer addSublayer:previewLayer];
  [session startRunning];

  if (error) {
    NSString *title = [NSString stringWithFormat:@"Failed with error %d", (int)[error code]];
    UIAlertController *alertController =
        [UIAlertController alertControllerWithTitle:title
                                            message:[error localizedDescription]
                                     preferredStyle:UIAlertControllerStyleAlert];
    UIAlertAction *dismiss =
        [UIAlertAction actionWithTitle:@"Dismiss" style:UIAlertActionStyleDefault handler:nil];
    [alertController addAction:dismiss];
    [self presentViewController:alertController animated:YES completion:nil];
    [self teardownAVCapture];
  }
   self.navigationController.navigationBar.hidden = YES;

}

- (IBAction)btLoadClick:(id)sender {
    [session stopRunning];

    pauseu=true;
    UIImagePickerController *imagePickerController = [[UIImagePickerController alloc]init];
    
    imagePickerController.delegate = self;
    NSArray *mediaTypesAllowed = [UIImagePickerController availableMediaTypesForSourceType:UIImagePickerControllerSourceTypePhotoLibrary];
    [imagePickerController setMediaTypes:mediaTypesAllowed];
    

    [self presentModalViewController:imagePickerController animated:YES];

}

- (void)imagePickerControllerDidCancel:(UIImagePickerController *)picker {
    [self dismissModalViewControllerAnimated:YES];
   
    pauseu=false;
    [session startRunning];
}

- (CVPixelBufferRef) pixelBufferFromCGImage: (CGImageRef) image
{
    NSDictionary *options = @{
                              (NSString*)kCVPixelBufferCGImageCompatibilityKey : @YES,
                              (NSString*)kCVPixelBufferCGBitmapContextCompatibilityKey : @YES,
                              };
    
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, CGImageGetWidth(image),
                                          CGImageGetHeight(image), kCVPixelFormatType_32ARGB, (__bridge CFDictionaryRef) options,
                                          &pxbuffer);
    if (status!=kCVReturnSuccess) {
        NSLog(@"Operation failed");
    }
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);
    
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);

    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, CGImageGetWidth(image),
                                                 CGImageGetHeight(image), 8,
                                                 4*CGImageGetWidth(image),
                                                 rgbColorSpace,
                                                 kCGImageAlphaNoneSkipLast);//kCGImageAlphaNone) ;//kCGImageAlphaFirst);//kCGImageAlphaNoneSkipFirst);
                                               //  kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);//kCGImageAlphaNoneSkipFirst);
    NSParameterAssert(context);
    
    CGContextConcatCTM(context, CGAffineTransformMakeRotation(0));
    CGAffineTransform flipVertical = CGAffineTransformMake( 1, 0, 0, -1, 0, CGImageGetHeight(image) );
    CGContextConcatCTM(context, flipVertical);
    CGAffineTransform flipHorizontal = CGAffineTransformMake( -1.0, 0.0, 0.0, 1.0, CGImageGetWidth(image), 0.0 );
    CGContextConcatCTM(context, flipHorizontal);
    
    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image),
                                           CGImageGetHeight(image)), image);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    return pxbuffer;
}


- (UIImage*)rotateUIImage:(UIImage*)sourceImage clockwise:(BOOL)clockwise
{
    CGSize size = sourceImage.size;
    UIGraphicsBeginImageContext(CGSizeMake(size.height, size.width));
    [[UIImage imageWithCGImage:[sourceImage CGImage] scale:1.0 orientation:clockwise ? UIImageOrientationRight : UIImageOrientationLeft] drawInRect:CGRectMake(0,0,size.height ,size.width)];
    UIImage* newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return newImage;
}

- (void)imagePickerController:(UIImagePickerController *)picker
        didFinishPickingImage:(UIImage *)image
                  editingInfo:(NSDictionary *)info
{

    [picker dismissModalViewControllerAnimated:YES];

    imGallery.image=image;
   
    pauseu=false;

    imGallery.contentMode = UIViewContentModeScaleAspectFit;
  
    
    [imGallery setBackgroundColor:[UIColor blackColor]];
    if (image!=nil) {
       
        UIImage *rotImage=[self rotateUIImage:image clockwise:TRUE];
       
        CGImageRef img = rotImage.CGImage;
        CVPixelBufferRef pixelBuffer= [self pixelBufferFromCGImage:img];
        

        oldPredictionValues1=nil;
        oldPredictionValues2=nil;
        oldPredictionValues3=nil;
        oldPredictionValues1 = [[NSMutableDictionary alloc] init];
        oldPredictionValues2 = [[NSMutableDictionary alloc] init];
        oldPredictionValues3 = [[NSMutableDictionary alloc] init];
        [self runCNNOnFrame:pixelBuffer];
        CFRelease(pixelBuffer);
        previewLayer.hidden=YES;
        imGallery.hidden=NO;
        [btFreeze setTitle:@"Return Camera" forState:UIControlStateNormal];
    }

}
- (void)teardownAVCapture {
  [stillImageOutput removeObserver:self forKeyPath:@"isCapturingStillImage"];
  [previewLayer removeFromSuperlayer];
}
bool forward=false;

- (IBAction)btMedicmind:(id)sender {

    forward=true;
     self.navigationController.navigationBar.hidden =  NO;
    [self performSegueWithIdentifier:@"backToTable" sender:self];
    pauseu=true;

}


- (void)didMoveToParentViewController:(UIViewController *)parent
{
    if (!forward) {
        self.navigationController.navigationBar.hidden =  YES;
        pauseu=false;
    } else {
        self.navigationController.navigationBar.hidden =  NO;
    }
    forward=false;

}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context {
  if (context == AVCaptureStillImageIsCapturingStillImageContext) {
    BOOL isCapturingStillImage =
        [[change objectForKey:NSKeyValueChangeNewKey] boolValue];

    if (isCapturingStillImage) {
      // do flash bulb like animation
      flashView = [[UIView alloc] initWithFrame:[previewView frame]];
      [flashView setBackgroundColor:[UIColor whiteColor]];
      [flashView setAlpha:0.f];
      [[[self view] window] addSubview:flashView];

      [UIView animateWithDuration:.4f
                       animations:^{
                         [flashView setAlpha:1.f];
                       }];
    } else {
      [UIView animateWithDuration:.4f
          animations:^{
            [flashView setAlpha:0.f];
          }
          completion:^(BOOL finished) {
            [flashView removeFromSuperview];
            flashView = nil;
          }];
    }
  }
}

- (AVCaptureVideoOrientation)avOrientationForDeviceOrientation:
    (UIDeviceOrientation)deviceOrientation {
  AVCaptureVideoOrientation result =
      (AVCaptureVideoOrientation)(deviceOrientation);
  if (deviceOrientation == UIDeviceOrientationLandscapeLeft)
    result = AVCaptureVideoOrientationLandscapeRight;
  else if (deviceOrientation == UIDeviceOrientationLandscapeRight)
    result = AVCaptureVideoOrientationLandscapeLeft;
  return result;
}

- (IBAction)takePicture:(id)sender {
  if ([session isRunning]) {
    [session stopRunning];
    [sender setTitle:@"Continue" forState:UIControlStateNormal];

    flashView = [[UIView alloc] initWithFrame:[previewView frame]];
    [flashView setBackgroundColor:[UIColor whiteColor]];
    [flashView setAlpha:0.f];
    [[[self view] window] addSubview:flashView];

    [UIView animateWithDuration:.2f
        animations:^{
          [flashView setAlpha:1.f];
        }
        completion:^(BOOL finished) {
          [UIView animateWithDuration:.2f
              animations:^{
                [flashView setAlpha:0.f];
              }
              completion:^(BOOL finished) {
                [flashView removeFromSuperview];
                flashView = nil;
              }];
        }];

  } else {
      imGallery.hidden=YES;
      imGallery.image=nil;//vwimage=nil;
      pauseu=false;
      previewLayer.hidden=NO;
      oldPredictionValues1=nil;
      oldPredictionValues2=nil;
      oldPredictionValues3=nil;
      oldPredictionValues1 = [[NSMutableDictionary alloc] init];
      oldPredictionValues2 = [[NSMutableDictionary alloc] init];
      oldPredictionValues3 = [[NSMutableDictionary alloc] init];

    [session startRunning];
    [sender setTitle:@"Freeze Frame" forState:UIControlStateNormal];
  }
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    
    if (pauseu)
        return;
    CVPixelBufferRef pixelBuffer;
   
    pixelBuffer= CMSampleBufferGetImageBuffer(sampleBuffer);

    CFRetain(pixelBuffer);
    [self runCNNOnFrame:pixelBuffer];
    CFRelease(pixelBuffer);
}

- (void)runCNNOnFrame:(CVPixelBufferRef)pixelBuffer {

  assert(pixelBuffer != NULL);

  OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
  int doReverseChannels;
  if (kCVPixelFormatType_32ARGB == sourcePixelFormat) {
    doReverseChannels = 1;
  } else if (kCVPixelFormatType_32BGRA == sourcePixelFormat) {
    doReverseChannels = 0;
  } else {
    assert(false);  // Unknown source format
  }

  const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
  const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
  const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);

  CVPixelBufferLockFlags unlockFlags = kNilOptions;
  CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);

  unsigned char *sourceBaseAddr =
      (unsigned char *)(CVPixelBufferGetBaseAddress(pixelBuffer));
  int image_height;
  unsigned char *sourceStartAddr;
  if (fullHeight <= image_width) {
    image_height = fullHeight;
    sourceStartAddr = sourceBaseAddr;
  } else {
    image_height = image_width;
    const int marginY = ((fullHeight - image_width) / 2);
    sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
  }
    const int image_channels = 4;

  assert(image_channels >= wanted_input_channels);
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape(
          {1, wanted_input_height, wanted_input_width, wanted_input_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  tensorflow::uint8 *in = sourceStartAddr;
  float *out = image_tensor_mapped.data();
  for (int y = 0; y < wanted_input_height; ++y) {
    float *out_row = out + (y * wanted_input_width * wanted_input_channels);
    for (int x = 0; x < wanted_input_width; ++x) {
      const int in_x = (y * image_width) / wanted_input_width;
      const int in_y = (x * image_height) / wanted_input_height;
      tensorflow::uint8 *in_pixel =
          in + (in_y * image_width * image_channels) + (in_x * image_channels);
      float *out_pixel = out_row + (x * wanted_input_channels);
      for (int c = 0; c < wanted_input_channels; ++c) {
        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;

      }
    }
  }

  CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);

  if (tf_session.get()) {
    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = tf_session->Run(
        {{input_layer_name, image_tensor}}, {output_layer_name}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed:" << run_status;
    } else {
      tensorflow::Tensor *output = &outputs[0];
      auto predictions = output->flat<float>();

      NSMutableDictionary *newValues = [NSMutableDictionary dictionary];
      for (int index = 0; index < predictions.size(); index += 1) {
          const float predictionValue = predictions(index);
          printf("xx %f \n",predictionValue);

          std::string label = labels[index % predictions.size()];
          NSString *labelObject = [NSString stringWithUTF8String:label.c_str()];
          NSNumber *valueObject = [NSNumber numberWithFloat:predictionValue];
          [newValues setObject:valueObject forKey:labelObject];

      }
      dispatch_async(dispatch_get_main_queue(), ^(void) {
        [self setPredictionValues:newValues];
      });
    }
  }
  CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)dealloc {
  [self teardownAVCapture];
}

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
}

- (void)viewDidLoad {
  [super viewDidLoad];

  labelLayers = [[NSMutableArray alloc] init];
  oldPredictionValues1 = [[NSMutableDictionary alloc] init];
  oldPredictionValues2 = [[NSMutableDictionary alloc] init];
  oldPredictionValues3 = [[NSMutableDictionary alloc] init];

  tensorflow::Status load_status;
  if (model_uses_memory_mapping) {
    load_status = LoadMemoryMappedModel(
        model_file_name, model_file_type, &tf_session, &tf_memmapped_env);
  } else {
    load_status = LoadModel(model_file_name, model_file_type, &tf_session);
  }
  if (!load_status.ok()) {
    LOG(FATAL) << "Couldn't load model: " << load_status;
  }

  tensorflow::Status labels_status =
      LoadLabels(labels_file_name, labels_file_type, &labels);
  if (!labels_status.ok()) {
    LOG(FATAL) << "Couldn't load labels: " << labels_status;
  }
  [self setupAVCapture];
}

- (void)viewDidUnload {
  [super viewDidUnload];
}

- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
}

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:animated];
}

- (void)viewDidDisappear:(BOOL)animated {
  [super viewDidDisappear:animated];
}

- (BOOL)shouldAutorotateToInterfaceOrientation:
    (UIInterfaceOrientation)interfaceOrientation {
  return (interfaceOrientation == UIInterfaceOrientationPortrait);
}

- (BOOL)prefersStatusBarHidden {
  return YES;
}

- (void)setPredictionValues:(NSDictionary *)newValues {

  for (NSString *label in newValues) {
    NSNumber *newPredictionValueObject = [newValues objectForKey:label];
    NSNumber *oldPredictionValueObject1 =[oldPredictionValues1 objectForKey:label];
    if (!oldPredictionValueObject1) {
      oldPredictionValueObject1 = [NSNumber numberWithFloat:-999.0f];
    }
    NSNumber *oldPredictionValueObject2 =[oldPredictionValues2 objectForKey:label];
    if (!oldPredictionValueObject2) {
          oldPredictionValueObject2 = [NSNumber numberWithFloat:-999.0f];
    }
    NSNumber *oldPredictionValueObject3 =[oldPredictionValues3 objectForKey:label];
    if (!oldPredictionValueObject3) {
          oldPredictionValueObject3 = [NSNumber numberWithFloat:-999.0f];
    }
      
    float newPredictionValue = [newPredictionValueObject floatValue];

    if (newPredictionValue<0)
        newPredictionValue=0;
    else if (newPredictionValue>1)
        newPredictionValue=1;
    newPredictionValueObject =[NSNumber numberWithFloat:newPredictionValue];

    [oldPredictionValues3 setObject:oldPredictionValueObject2 forKey:label];
    [oldPredictionValues2 setObject:oldPredictionValueObject1 forKey:label];
    [oldPredictionValues1 setObject:newPredictionValueObject forKey:label];

  }

  NSArray *candidateLabels = [NSMutableArray array];
  for (NSString *label in oldPredictionValues1) {
    float sumPredictionValue=0;
    int cnt=0;
    NSNumber *oldPredictionValueObject1 =  [oldPredictionValues1 objectForKey:label];
    float oldPredictionValue1 = [oldPredictionValueObject1 floatValue];
    if (oldPredictionValue1>0) {
      sumPredictionValue+= [oldPredictionValueObject1 floatValue];
      cnt++;
    }

    NSNumber *oldPredictionValueObject2 =  [oldPredictionValues2 objectForKey:label];
    float oldPredictionValue2 = [oldPredictionValueObject2 floatValue];
    if (oldPredictionValue2>0) {
          sumPredictionValue+= [oldPredictionValueObject2 floatValue];
          cnt++;
    }
    NSNumber *oldPredictionValueObject3 =  [oldPredictionValues3 objectForKey:label];
    float oldPredictionValue3 = [oldPredictionValueObject3 floatValue];
    if (oldPredictionValue3>0) {
          sumPredictionValue+= [oldPredictionValueObject3 floatValue];
          cnt++;
    }
    float mean;
    if (cnt>0) {
      mean=sumPredictionValue/cnt;
      if (mean<0.2)
          mean=0;
      else if (mean>1)
          mean=1;
    } else
        mean=0;
    NSNumber *maxPredictionValueObject =[NSNumber numberWithFloat:mean];
      
      NSDictionary *entry = @{
        @"label" : label,
        @"value" : maxPredictionValueObject
      };
      candidateLabels = [candidateLabels arrayByAddingObject:entry];
   // }
  }
  NSSortDescriptor *sort =
      [NSSortDescriptor sortDescriptorWithKey:@"value" ascending:NO];
  NSArray *sortedLabels = [candidateLabels
      sortedArrayUsingDescriptors:[NSArray arrayWithObject:sort]];

  const float leftMargin = 10.0f;
  const float topMargin = 10.0f;

  const float valueWidth = 88.0f;
  const float valueHeight = 26.0f;

  const float labelWidth = 246.0f;
  const float labelHeight = 26.0f;

  const float labelMarginX = 5.0f;
  const float labelMarginY = 5.0f;

  [self removeAllLabelLayers];

  int labelCount = 0;
  for (NSDictionary *entry in sortedLabels) {
      NSString *label = @"Pneumonia";//[entry objectForKey:@"label"];
    NSNumber *valueObject = [entry objectForKey:@"value"];
    const float value = [valueObject floatValue];

    const float originY =
        (topMargin + ((labelHeight + labelMarginY) * labelCount));

    const float valueOriginX = leftMargin;
      NSString *valueText = [NSString stringWithFormat:@"%.2f", value];

    [self addLabelLayerWithText:valueText
                        originX:valueOriginX
                        originY:originY
                          width:valueWidth
                         height:valueHeight
                      alignment:kCAAlignmentRight];

    const float labelOriginX = (leftMargin + valueWidth + labelMarginX);

    [self addLabelLayerWithText:[label capitalizedString]
                        originX:labelOriginX
                        originY:originY
                          width:labelWidth
                         height:labelHeight
                      alignment:kCAAlignmentLeft];


    labelCount += 1;
    if (labelCount > 4) {
      break;
    }
  }
}

- (void)removeAllLabelLayers {
  for (CATextLayer *layer in labelLayers) {
    [layer removeFromSuperlayer];
  }
  [labelLayers removeAllObjects];
}

- (void)addLabelLayerWithText:(NSString *)text
                      originX:(float)originX
                      originY:(float)originY
                        width:(float)width
                       height:(float)height
                    alignment:(NSString *)alignment {
  CFTypeRef font = (CFTypeRef) @"Menlo-Regular";
  const float fontSize = 20.0f;

  const float marginSizeX = 5.0f;
  const float marginSizeY = 2.0f;

  const CGRect backgroundBounds = CGRectMake(originX, originY, width, height);

  const CGRect textBounds =
      CGRectMake((originX + marginSizeX), (originY + marginSizeY),
                 (width - (marginSizeX * 2)), (height - (marginSizeY * 2)));

  CATextLayer *background = [CATextLayer layer];
  [background setBackgroundColor:[UIColor blackColor].CGColor];
  [background setOpacity:0.5f];
  [background setFrame:backgroundBounds];
  background.cornerRadius = 5.0f;

  [[self.view layer] addSublayer:background];
  [labelLayers addObject:background];

  CATextLayer *layer = [CATextLayer layer];
  [layer setForegroundColor:[UIColor whiteColor].CGColor];
  [layer setFrame:textBounds];
  [layer setAlignmentMode:alignment];
  [layer setWrapped:YES];
  [layer setFont:font];
  [layer setFontSize:fontSize];
  layer.contentsScale = [[UIScreen mainScreen] scale];
  [layer setString:text];

  [[self.view layer] addSublayer:layer];
  [labelLayers addObject:layer];
}

- (void)setPredictionText:(NSString *)text withDuration:(float)duration {
  if (duration > 0.0) {
    CABasicAnimation *colorAnimation =
        [CABasicAnimation animationWithKeyPath:@"foregroundColor"];
    colorAnimation.duration = duration;
    colorAnimation.fillMode = kCAFillModeForwards;
    colorAnimation.removedOnCompletion = NO;
    colorAnimation.fromValue = (id)[UIColor darkGrayColor].CGColor;
    colorAnimation.toValue = (id)[UIColor whiteColor].CGColor;
    colorAnimation.timingFunction =
        [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionLinear];
    [self.predictionTextLayer addAnimation:colorAnimation
                                    forKey:@"colorAnimation"];
  } else {
    self.predictionTextLayer.foregroundColor = [UIColor whiteColor].CGColor;
  }

  [self.predictionTextLayer removeFromSuperlayer];
  [[self.view layer] addSublayer:self.predictionTextLayer];
  [self.predictionTextLayer setString:text];
}

@end
