"use client";

import {
  useFrcnnNoAttack,
  useFrcnnWithAttack,
  useSddNoAttack,
  useSddWithAttack,
  useYolov9NoAttack,
  useYolov9WithAttack,
  useYolov8,
} from "@/hooks";
import { ChangeEvent, FormEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function Home() {
  const defaultThreshold = [0.25];
  const [image, setImage] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [detectedImageUrl, setDetectedImageUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(defaultThreshold[0]);
  const [error, setError] = useState<string | null>(null);

  const yolov9_with_attack = useYolov9WithAttack(image, threshold);
  const yolov9_no_attack = useYolov9NoAttack(image, threshold);
  const yolov8 = useYolov8(image, threshold);

  const frcnn_with_attack = useFrcnnWithAttack(image, threshold);
  const frcnn_no_attack = useFrcnnNoAttack(image, threshold);

  const ssd_with_attack = useSddWithAttack(image, threshold);
  const ssd_no_attack = useSddNoAttack(image, threshold);

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  const onSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!image) {
      console.error("No image selected");
      return;
    }

    setDetectedImageUrl(null); // Reset detected image URL
    setError(null); // Reset error

    if (selectedModel === "yolov9_with_attack" && image) {
      yolov9_with_attack.mutate(undefined, {
        onSuccess: (data) => setDetectedImageUrl(data), // Handle successful response
        onError: (err) => setError(err.message), // Handle error response
      });
    } else if (selectedModel === "yolov9_no_attack" && image) {
      yolov9_no_attack.mutate(undefined, {
        onSuccess: (data) => setDetectedImageUrl(data),
        onError: (err) => setError(err.message),
      });
    } else if (selectedModel === "yolov8" && image) {
      yolov8.mutate(undefined, {
        onSuccess: (data) => setDetectedImageUrl(data),
        onError: (err) => setError(err.message),
      });
    } else if (selectedModel === "frcnn_with_attack" && image) {
      frcnn_with_attack.mutate(undefined, {
        onSuccess: (data) => setDetectedImageUrl(data),
        onError: (err) => setError(err.message),
      });
    } else if (selectedModel === "frcnn_no_attack" && image) {
      frcnn_no_attack.mutate(undefined, {
        onSuccess: (data) => setDetectedImageUrl(data),
        onError: (err) => setError(err.message),
      });
    } else if (selectedModel === "ssd_with_attack" && image) {
      ssd_with_attack.mutate(undefined, {
        onSuccess: (data) => setDetectedImageUrl(data),
        onError: (err) => setError(err.message),
      });
    } else if (selectedModel === "ssd_no_attack" && image) {
      ssd_no_attack.mutate(undefined, {
        onSuccess: (data) => setDetectedImageUrl(data),
        onError: (err) => setError(err.message),
      });
    } else {
      console.error("No model selected or unsupported model selected");
    }
  };

  const handleModelSelect = (value: string) => {
    setSelectedModel(value);
  };

  const isPending =
    yolov9_with_attack.isPending ||
    yolov9_no_attack.isPending ||
    frcnn_with_attack.isPending ||
    frcnn_no_attack.isPending ||
    ssd_with_attack.isPending ||
    ssd_no_attack.isPending ||
    yolov8.isPending;

  return (
    <div className="max-w-5xl mx-auto p-10 justify-center items-center flex flex-col">
      <form
        onSubmit={onSubmit}
        className="flex flex-col justify-center items-center gap-5 w-full"
      >
        <div className="grid w-full items-center gap-3">
          <Label htmlFor="select">Select model</Label>
          <Select onValueChange={handleModelSelect}>
            <SelectTrigger id="select" className="w-full">
              <SelectValue placeholder="Pick a model" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value="yolov8">Yolov8</SelectItem>
                <SelectItem value="yolov9_no_attack">
                  Yolov9 Without Adversarial Attack
                </SelectItem>
                <SelectItem value="yolov9_with_attack">
                  Yolov9 With Adversarial Attack
                </SelectItem>
                <SelectItem value="frcnn_no_attack">
                  F-R-CNN Without Adversarial Attack
                </SelectItem>
                <SelectItem value="frcnn_with_attack">
                  F-R-CNN With Adversarial Attack
                </SelectItem>
                <SelectItem value="ssd_no_attack">
                  SSD Without Adversarial Attack
                </SelectItem>
                <SelectItem value="ssd_with_attack">
                  SSD With Adversarial Attack
                </SelectItem>
                <SelectItem value="all">All</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>

        <div className="grid w-full items-center gap-3">
          <Label htmlFor="threshold">Score Percentage</Label>
          <div className="flex flex-row gap-3 justify-center items-center">
            <Slider
              defaultValue={defaultThreshold}
              max={1}
              step={0.01}
              onValueChange={(value) => setThreshold(value[0])}
            />
            <p className="text-sm font-black text-red-600">{threshold}</p>
          </div>
        </div>

        <div className="grid w-full items-center gap-3">
          <Label htmlFor="picture">
            Upload your picture or video to predict
          </Label>
          <Input id="picture" type="file" onChange={onChange} />
        </div>

        <Button
          type="submit"
          disabled={!image || !selectedModel}
          className="w-full"
        >
          {isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Please wait
            </>
          ) : (
            "Submit"
          )}
        </Button>
      </form>

      {error && <p>Error uploading image: {error}</p>}

      {isPending && (
        <div className="flex flex-col space-y-3 mt-10 w-full">
          <Skeleton className="md:h-[500px] h-[200px] rounded-xl w-full" />
        </div>
      )}

      {detectedImageUrl && (
        <div className="mt-8 gap-5 flex flex-col">
          <h3>Detected Image:</h3>
          <img src={detectedImageUrl} alt="Detected" className="rounded-xl" />
        </div>
      )}
    </div>
  );
}
