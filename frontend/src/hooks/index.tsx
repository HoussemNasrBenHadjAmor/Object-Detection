import { getPredictionYolov9, getPredictionFRCNN } from "@/api";
import { useMutation } from "@tanstack/react-query";

const useYolov8 = () => {
  return useMutation({
    mutationFn: getPredictionYolov9,
  });
};

const useYolov9 = () => {
  return useMutation({
    mutationFn: getPredictionYolov9,
  });
};

const useYolov10 = () => {
  return useMutation({
    mutationFn: getPredictionYolov9,
  });
};

const useFrcnn = () => {
  return useMutation({
    mutationFn: getPredictionFRCNN,
  });
};

export { useYolov9, useYolov10, useYolov8, useFrcnn };
