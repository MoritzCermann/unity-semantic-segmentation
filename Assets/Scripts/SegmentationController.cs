using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class SegmentationController : MonoBehaviour
{
    private RunUnet runUnet;
    private ModelInput modelInput;

    private Texture2D arTexture;

    [SerializeField]
    private RawImage arImage;
    [SerializeField]
    private RawImage semanticImage;

    [SerializeField]
    private TextMeshProUGUI inferenceTimeText;
    [SerializeField]
    private TextMeshProUGUI downloadTimeText;
    [SerializeField]
    private TextMeshProUGUI drawTimeText;
    [SerializeField]
    private TextMeshProUGUI totalTime;

    [SerializeField]
    private Transform classLabelPanel;
    [SerializeField]
    private GameObject classLabelPrefab;

    [SerializeField]
    private Slider slider;
    [SerializeField]
    private Button playPauseBtn;
    [SerializeField]
    private TextMeshProUGUI displayFPS;

    private float inferenceInterval = 0.19f; // execute inference every 0.19 seconds >> 5 fps
    private float nextInferenceTime = 0.0f;

    private bool finishedInference = true;
    private bool runInferenceOnLoop = false;

    private void Start()
    {
        runUnet = FindObjectOfType<RunUnet>();
        modelInput = FindObjectOfType<ModelInput>();

        slider.onValueChanged.AddListener(OnSliderValueChanged);

        CreateClassLabelUI();

        OnDropdownChangedImgType(0); // default to Cutout
    }

    private void Update()
    {
        if (finishedInference && runInferenceOnLoop && Time.time >= nextInferenceTime)
        {
            nextInferenceTime = Time.time + inferenceInterval;

            RunSemanticSegmentation();
        }
    }

    public void RunSemanticSegmentation()
    {
        finishedInference = false;

        arTexture = modelInput.GetCameraTexture();

        arImage.rectTransform.sizeDelta = new Vector2(arTexture.width, arTexture.height);
        arImage.texture = arTexture;

        Texture2D outputTexture = runUnet.ProcessImage(arTexture);
        Debug.Log($"Output Texture: {outputTexture.width} - {outputTexture.height}");


        DisplayTime();

        semanticImage.texture = outputTexture;

        finishedInference = true;
    }

    private void DisplayTime()
    {
        long[] time = runUnet.GetTime();

        inferenceTimeText.text = time[0].ToString();
        //downloadTimeText.text = time[1].ToString();
        drawTimeText.text = time[2].ToString();

        totalTime.text = (time[0] /* + time[1] */ + time[2]).ToString();
    }

    public void OnDropdownChangedImgType(int index)
    {
        switch (index)
        {
            case 0:
                modelInput.ChangeImgType(ImageType.Cutout);
                break;
            case 1:
                modelInput.ChangeImgType(ImageType.Scaled);
                break;
        }       
    }

    public void OnButtonPlayPause()
    {
        runInferenceOnLoop = !runInferenceOnLoop;

        playPauseBtn.GetComponentInChildren<TextMeshProUGUI>().text = runInferenceOnLoop ? "Pause" : "Play";
        if (runInferenceOnLoop)
            displayFPS.enabled = true;
        else
            displayFPS.enabled = false;
        Debug.Log(runInferenceOnLoop);
    }

    private void OnSliderValueChanged(float value)
    {
        Debug.Log(value);
        Color color = semanticImage.color;
        color.a = value;
        semanticImage.color = color;
    }

    /// <summary>
    /// Creates a label for each class in the model
    /// Will update when classes or their color is changed
    /// </summary>
    private void CreateClassLabelUI()
    {
        var classes = runUnet.GetClasses();

        foreach (SemanticClass semanticClass in classes)
        {
            GameObject label = Instantiate(classLabelPrefab, classLabelPanel);

            label.GetComponentInChildren<TextMeshProUGUI>().text = semanticClass.name;
            label.GetComponentInChildren<Image>().color = semanticClass.color;
        }
    }
}
