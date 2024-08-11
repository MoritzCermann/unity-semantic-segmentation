using UnityEngine;
using UnityEngine.UI;
using Unity.Sentis;
using Unity.Sentis.Layers;
using System.Collections.Generic;
using System;
using System.Collections;

public class RunUnet : MonoBehaviour
{
    public ModelAsset unetDefault;
    public ModelAsset unetMini;

    private IWorker worker;

    /// <summary>
    /// The input shape is defined by the model and will be automatically set when the model is loaded or changed.
    /// Should consist of Height, Width, and Channel each as an integer.
    /// </summary>
    private int[] inputShape;

    private long inferenceTime;
    private long downloadTime;
    private long drawTime;


    private const int numClasses = 11;
    [SerializeField]
    private SemanticClass[] classes = new SemanticClass[12]; // includes void class

    public Action modelHasChanged;

    void Start()
    {
        StartCoroutine(Init());
    }

    private IEnumerator Init()
    {
        yield return new WaitForSeconds(0.5f);
        LoadModel(unetMini);
    }

    public void LoadModel(ModelAsset modelAsset)
    {
        if (worker != null)
        {
            worker.Dispose();
        }

        var model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.CPU, model);

        SymbolicTensorShape tensorInputShape = model.inputs[0].shape;       
        if (tensorInputShape[1].isValue && tensorInputShape[2].isValue && tensorInputShape[3].isValue)
        {
            inputShape = new int[] { tensorInputShape[1].value, tensorInputShape[2].value, tensorInputShape[3].value };
            Debug.Log($"Model shape >> Height: {inputShape[0]}, Width: {inputShape[1]}, Channel: {inputShape[2]}");
        }
        else
        {
            throw new InvalidOperationException("Unsupported tensor dimension. Check your Model");
        }

        modelHasChanged?.Invoke();
    }

    public Texture2D ProcessImage(Texture2D inputTexture)
    {
        System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

        int height= inputShape[0];
        int width = inputShape[1];
        int channels = inputShape[2];

        TextureTransform settings = new TextureTransform().SetDimensions(height, width, channels).SetTensorLayout(TensorLayout.NHWC);

        using var inputTensor = TextureConverter.ToTensor(inputTexture, settings);
        Debug.Log(inputTensor.dataOnBackend.backendType); // Check if the tensor is stored in CPU or GPU memory.

        worker.Execute(inputTensor);
     
        TensorFloat outputTensor = worker.PeekOutput() as TensorFloat;
        outputTensor.CompleteOperationsAndDownload(); // SUPER SLOW ???!!!
        float[] outputData = outputTensor.ToReadOnlyArray(); // SUPER SLOW ???!!!

        stopwatch.Stop();
        inferenceTime = stopwatch.ElapsedMilliseconds;
        stopwatch.Restart();

        outputTensor.Dispose();

        //stopwatch.Stop();
        //downloadTime = stopwatch.ElapsedMilliseconds;
        //stopwatch.Restart();

        Texture2D outputTexture = OutputToTexture(outputData, height, width);
        stopwatch.Stop();
        drawTime = stopwatch.ElapsedMilliseconds;

        return outputTexture;
    }

    Texture2D OutputToTexture(float[] outputData, int height, int width)
    {
        var classMap = new int[width, height];


        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float maxPred = float.MinValue;
                int maxIndex = 0;
                for (int c = 0; c < numClasses; c++)
                {
                    float pred = outputData[y * width * numClasses + x * numClasses + c];

                    if (pred > maxPred)
                    {
                        maxPred = pred;
                        maxIndex = c;
                    }
                }
                classMap[x, y] = maxIndex;
            }
        }

        Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Color32 color = GetColor(classMap[x, y]);

                texture.SetPixel(x, height - 1 - y, color);
            }
        }
        texture.Apply();

        return texture;
    }

    public void OnDropdownChangedModel(int index)
    {
        switch (index)
        {
            case 0:
                LoadModel(unetDefault);
                break;
            case 1:
                LoadModel(unetMini);
                break;
        }
    }

    Color GetColor(int classIndex)
    {
        switch (classIndex)
        {
            case 0: return classes[0].color; // Sky
            case 1: return classes[1].color; // Building 
            case 2: return classes[2].color; // Pole
            case 3: return classes[3].color; // Road
            case 4: return classes[4].color; // Pavement
            case 5: return classes[5].color; // Tree
            case 6: return classes[6].color; // SignSymbol
            case 7: return classes[7].color; // Fence
            case 8: return classes[8].color; // Car
            case 9: return classes[9].color; // Pedestrian
            case 10: return classes[10].color; // Bicyclist
            default: return classes[11].color; ; // Void
        }
    }

    public long[] GetTime()
    {
        return new long[] { inferenceTime, downloadTime, drawTime };
    }

    public SemanticClass[] GetClasses()
    {
        return classes;
    }

    public int[] GetInputShape()
    {
          return inputShape;
    } 

    void OnDestroy()
    {
        worker.Dispose();
    }
}


[Serializable]
public struct SemanticClass
{
    public string name;
    public Color color;
}