export const getPredictionYolov9 = async (image: File | null) => {
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const res = await fetch("http://127.0.0.1:8000/detect/yolov9", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  return url;
};

export const getPredictionYolov8 = async (image: File | null) => {
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const res = await fetch("http://127.0.0.1:8000/detect/", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  return url;
};

export const getPredictionYolov10 = async (image: File | null) => {
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const res = await fetch("http://127.0.0.1:8000/detect/", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  return url;
};

export const getPredictionFRCNN = async (image: File | null) => {
  if (!image) {
    throw new Error("No image provided");
  }

  const formData = new FormData();
  formData.append("file", image);

  const res = await fetch("http://127.0.0.1:8000/detect/frcnn", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Error uploading image");
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  return url;
};
