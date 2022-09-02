import 'package:camera/camera.dart';
import 'package:image/image.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/src/image/color_space_type.dart';
import 'package:tflite_flutter_helper/src/tensorbuffer/tensorbuffer.dart';

/// Implements some stateless image conversion methods.
///
/// This class is an internal helper.
class ImageConversions {
  static Image convertRgbTensorBufferToImage(TensorBuffer buffer) {
    List<int> shape = buffer.getShape();
    ColorSpaceType rgb = ColorSpaceType.RGB;
    rgb.assertShape(shape);

    int h = rgb.getHeight(shape);
    int w = rgb.getWidth(shape);
    Image image = Image(w, h);

    List<int> rgbValues = buffer.getIntList();
    assert(rgbValues.length == w * h * 3);

    for (int i = 0, j = 0, wi = 0, hi = 0; j < rgbValues.length; i++) {
      int r = rgbValues[j++];
      int g = rgbValues[j++];
      int b = rgbValues[j++];
      image.setPixelRgba(wi, hi, r, g, b);
      wi++;
      if (wi % w == 0) {
        wi = 0;
        hi++;
      }
    }

    return image;
  }

  static Image convertGrayscaleTensorBufferToImage(TensorBuffer buffer) {
    // Convert buffer into Uint8 as needed.
    TensorBuffer uint8Buffer = buffer.getDataType() == TfLiteType.uint8
        ? buffer
        : TensorBuffer.createFrom(buffer, TfLiteType.uint8);

    final shape = uint8Buffer.getShape();
    final grayscale = ColorSpaceType.GRAYSCALE;
    grayscale.assertShape(shape);

    final image = Image.fromBytes(grayscale.getWidth(shape),
        grayscale.getHeight(shape), uint8Buffer.getBuffer().asUint8List(),
        format: Format.luminance);

    return image;
  }

  static void convertImageToTensorBuffer(Image image, TensorBuffer buffer) {
    int w = image.width;
    int h = image.height;
    List<int> intValues = image.data;
    int flatSize = w * h * 3;
    List<int> shape = [h, w, 3];
    switch (buffer.getDataType()) {
      case TfLiteType.uint8:
        List<int> byteArr = List.filled(flatSize, 0);
        for (int i = 0, j = 0; i < intValues.length; i++) {
          byteArr[j++] = ((intValues[i]) & 0xFF);
          byteArr[j++] = ((intValues[i] >> 8) & 0xFF);
          byteArr[j++] = ((intValues[i] >> 16) & 0xFF);
        }
        buffer.loadList(byteArr, shape: shape);
        break;
      case TfLiteType.float32:
        List<double> floatArr = List.filled(flatSize, 0.0);
        for (int i = 0, j = 0; i < intValues.length; i++) {
          floatArr[j++] = ((intValues[i]) & 0xFF).toDouble();
          floatArr[j++] = ((intValues[i] >> 8) & 0xFF).toDouble();
          floatArr[j++] = ((intValues[i] >> 16) & 0xFF).toDouble();
        }
        buffer.loadList(floatArr, shape: shape);
        break;
      default:
        throw StateError(
            "${buffer.getDataType()} is unsupported with TensorBuffer.");
    }
  }

  static List<int>? convertImageToPng(CameraImage image) {
    Image img;
    if (image.format.group == ImageFormatGroup.yuv420) {
      img = convertYUV420(image);
    } else if (image.format.group == ImageFormatGroup.bgra8888) {
      img = convertBGRA8888(image);
    } else {
      return null;
    }

    PngEncoder pngEncoder = new PngEncoder();

    List<int> png = pngEncoder.encodeImage(img);
    return png;
  }

  static Image convertBGRA8888(CameraImage image) {
    return Image.fromBytes(
      image.width,
      image.height,
      image.planes[0].bytes,
      format: Format.bgra,
    );
  }

  static Image convertYUV420(CameraImage image) {
    var img = Image(image.width, image.height);

    Plane plane = image.planes[0];
    const int shift = (0xFF << 24);

    for (int x = 0; x < image.width; x++) {
      for (int planeOffset = 0;
          planeOffset < image.height * image.width;
          planeOffset += image.width) {
        final pixelColor = plane.bytes[planeOffset + x];
        var newVal =
            shift | (pixelColor << 16) | (pixelColor << 8) | pixelColor;
        img.data[planeOffset + x] = newVal;
      }
    }

    return img;
  }
}
