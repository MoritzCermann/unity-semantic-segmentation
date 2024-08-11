using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using static UnityEngine.XR.ARSubsystems.XRCpuImage;

public class ModelInput : MonoBehaviour
{
    private ARCameraManager arCameraManager;
    private RunUnet runUnet;

    /// <summary>
    /// The input shape is defined by the model and will be automatically set when the model is loaded or changed.
    /// Should consist of Height, Width, and Channel each as an integer.
    /// </summary>
    private int[] inputShape = new int[3];


    public ImageType imgType;
    

    private static readonly Vector2Int modelInput = new Vector2Int(256, 256);

    private void Start()
    {
        runUnet = FindObjectOfType<RunUnet>();
        arCameraManager = FindObjectOfType<ARCameraManager>();

        imgType = ImageType.Scaled;

        runUnet.modelHasChanged += OnModelChanged;
    }

    public Texture2D GetCameraTexture()
    {
        if (!arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage image)) // 480 x 640
            return null; // Dangerous, change later!!!

        var conversionParams = ConfigureConversion(image);

        Texture2D arTexture = new Texture2D(inputShape[1], inputShape[0], TextureFormat.RGBA32, false);
        var rawTextureData = arTexture.GetRawTextureData<byte>();

        // Converts XRCPUImage to Texture2D
        try
        {
            unsafe
            {
                image.Convert(
                    conversionParams,
                    new IntPtr(rawTextureData.GetUnsafePtr()),
                    rawTextureData.Length);
            }
        }
        finally
        {
            image.Dispose();
        }
        arTexture.Apply();
        arTexture = RotateTexture(arTexture, true);

        Debug.Log($"ImageType: {imgType}, Dimensions: {arTexture.width} - {arTexture.height}");

        return arTexture;
    }

    private void OnModelChanged()
    {
        inputShape = runUnet.GetInputShape();
    }

    public void ChangeImgType(ImageType imageType)
    {
        imgType = imageType;
    }

    private ConversionParams ConfigureConversion(XRCpuImage image)
    {
        ConversionParams conversionParams;
        int modelInputH = inputShape[0];
        int modelInputW = inputShape[1];


        int midH = image.height / 2 - modelInputH / 2;
        int midW = image.width / 2 - modelInputW / 2;

        switch (imgType)
        {
            case ImageType.Cutout:             
                conversionParams = new XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(midW, midH, modelInputW, modelInputH), // This will cut out the upper left 256x256 pixels
                    outputDimensions = new Vector2Int(modelInputW, modelInputH),
                    outputFormat = TextureFormat.RGBA32,
                    transformation = XRCpuImage.Transformation.MirrorX
                };
                break;

            case ImageType.Scaled:
                conversionParams = new XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(0, 0, image.width, image.height), // This will scale the image down to 256x256
                    outputDimensions = new Vector2Int(modelInputW, modelInputH),
                    outputFormat = TextureFormat.RGBA32,
                    transformation = XRCpuImage.Transformation.MirrorX
                };
                break;

            default:
                conversionParams = new XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(midW, midH, modelInputW, modelInputH), // This will cut out the upper left 256x256 pixels
                    outputDimensions = new Vector2Int(modelInputW, modelInputH),
                    outputFormat = TextureFormat.RGBA32,
                    transformation = XRCpuImage.Transformation.MirrorX
                };
                break;
        }
        return conversionParams;
    }

    /// <summary>
    /// Rotates a given texture by 90 degrees. Nessesary because XRCpuImage is recieved in portrait mode
    /// </summary>
    /// <param name="originalTexture"></param>
    /// <param name="clockwise">Wheter or not to rotate the texture clockwise. </param>
    /// <returns>90 degree rotated texture.</returns>
    private Texture2D RotateTexture(Texture2D originalTexture, bool clockwise)
    {
        Color32[] original = originalTexture.GetPixels32();
        Color32[] rotated = new Color32[original.Length];
        int w = originalTexture.width;
        int h = originalTexture.height;

        int iRotated, iOriginal;

        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                iRotated = (i + 1) * h - j - 1;
                iOriginal = clockwise ? original.Length - 1 - (j * w + i) : j * w + i;
                rotated[iRotated] = original[iOriginal];
            }
        }

        Texture2D rotatedTexture = new Texture2D(h, w);
        rotatedTexture.SetPixels32(rotated);
        rotatedTexture.Apply();
        return rotatedTexture;
    }

    void OnDestroy()
    {
        runUnet.modelHasChanged -= OnModelChanged;
    }
}


/// <summary>
/// Either returns the camera images as a scaled down 256x256 image or cuts out a 256x256 image from the center of the camera image which is not scaled
/// </summary>
public enum ImageType
{
    Cutout,
    Scaled
};