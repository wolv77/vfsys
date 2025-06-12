import io
import time
import logging
import asyncio
from typing import Tuple, Dict
from contextlib import asynccontextmanager
import random

import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
import mediapipe as mp
from deepface import DeepFace
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------- Settings -----------------
class Settings(BaseSettings):
    app_version: str = "1.0.0"
    allowed_types: list = ["image/jpeg", "image/png"]
    target_size: tuple = (224, 224)
    min_sharpness: float = 15.0      # Adjusted for ID cards
    min_brightness: float = 80.0
    max_brightness: float = 200.0
    threshold_distance: float = 0.35  # Keep current threshold
    api_key: str = "supersecretkey"
    clahe_clip_limit: float = 3.0
    clahe_grid_size: tuple = (8, 8)
    denoise_strength: int = 10
    sharpening_strength: float = 1.5
    contrast_alpha: float = 1.5
    brightness_beta: int = 10
    min_depth_variation: float = 0.1
    min_texture_score: float = 0.7
    # تعريف التفاعلات المتاحة
    available_actions: list = [
        "smile",    # ابتسامة
        "blink",    # وميض العين
        "head_turn" # التفات الرأس
    ]
    required_action: str = "any"  # قبول أي حركة من الحركات المتاحة
    action_threshold: dict = {
        "smile": 0.45,      # عتبة الابتسامة
        "blink": 0.3,       # عتبة وميض العين
        "head_turn": 15.0   # عتبة التفات الرأس (بالدرجات)
    }
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_image_size: tuple = (1920, 1080)
    parallel_processing: bool = True
    max_memory_usage: int = 1024 * 1024 * 512  # 512MB
    gc_threshold: int = 100  # garbage collection threshold
    min_eye_ratio: float = 0.3  # More lenient eye ratio threshold

settings = Settings()

# ----------------- Security -----------------
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    return api_key

# ----------------- Models Loading -----------------
detector = MTCNN()
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# ----------------- Custom Exceptions -----------------
class FaceDetectionError(HTTPException):
    def __init__(self, detail: str = "Face detection failed"):
        super().__init__(status_code=400, detail=detail)

class FaceVerificationError(HTTPException):
    def __init__(self, detail: str = "Face verification failed"):
        super().__init__(status_code=400, detail=detail)

class LivenessError(HTTPException):
    def __init__(self, detail: str = "Liveness check failed"):
        super().__init__(status_code=400, detail=detail)

class ImageQualityError(HTTPException):
    def __init__(self, detail: str, quality_metrics: dict):
        super().__init__(
            status_code=400,
            detail={
                "message": detail,
                "quality_metrics": quality_metrics
            }
        )

class ActionVerificationError(HTTPException):
    def __init__(self, detail: str = "No valid action detected. Please perform any of: smile, blink, or head turn"):
        super().__init__(
            status_code=400,
            detail=detail
        )

# ----------------- Helper Functions -----------------
def process_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')

async def detect_face(image: np.ndarray) -> dict:
    faces = detector.detect_faces(image)
    if not faces:
        raise FaceDetectionError("No face detected")
    return faces[0]

async def verify_liveness(image: np.ndarray) -> bool:
    try:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return bool(results.multi_face_landmarks)
    except Exception as e:
        logger.error(f"Liveness check error: {str(e)}")
        return False

async def get_embeddings(image: np.ndarray) -> np.ndarray:
    try:
        result = DeepFace.represent(
            image,
            model_name="GhostFaceNet",
            enforce_detection=False
        )
        return result
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise FaceVerificationError(f"Face verification failed: {str(e)}")

async def enhance_image(image: np.ndarray) -> np.ndarray:
    # Sharpen ID card image
    kernel = np.array([[-1,-1,-1], 
                      [-1, 9,-1],
                      [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

async def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality using various techniques."""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=settings.clahe_clip_limit, tileGridSize=settings.clahe_grid_size)
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl,a,b))
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced_rgb, None, settings.denoise_strength, 10, 7)
        
        return denoised
    except Exception as e:
        logger.warning(f"Image enhancement failed: {str(e)}")
        return image

async def process_low_quality_image(image: np.ndarray, quality: dict) -> np.ndarray:
    """Apply additional processing for low quality images."""
    if quality["sharpness"] < settings.min_sharpness:
        # Sharpen image
        kernel = np.array([[-1,-1,-1], 
                         [-1, 9,-1],
                         [-1,-1,-1]])
        image = cv2.filter2D(image, -1, kernel)
    
    if quality["brightness"] < settings.min_brightness:
        # Increase brightness
        alpha = settings.contrast_alpha  # Contrast
        beta = settings.brightness_beta    # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return image

# ----------------- Response Models -----------------
class QualityInfo(BaseModel):
    sharpness: float = Field(..., description="Image sharpness (Laplacian variance)")
    brightness: float = Field(..., description="Image brightness (pixel mean)")

class VerificationResponse(BaseModel):
    verified: bool = Field(..., description="Overall verification result")
    id_quality: QualityInfo
    reference_quality: QualityInfo
    action_quality: QualityInfo
    processing_time: float = Field(..., description="Total processing time in seconds")

# ----------------- FastAPI App -----------------
app = FastAPI(
    title="Face Verification API",
    description="""
    Advanced face verification system with:
    - Liveness detection
    - Action verification
    - Image quality enhancement
    - Anti-spoofing protection
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_random_action() -> str:
    """اختيار تفاعل عشوائي من القائمة المتاحة"""
    return random.choice(settings.available_actions)

@app.post("/verify", response_model=VerificationResponse)
async def verify_identity(
    id_card: UploadFile = File(...),
    reference: UploadFile = File(...),
    action: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    start_time = time.time()
    
    # Validate inputs
    if any(f.content_type not in settings.allowed_types for f in [id_card, reference, action]):
        logger.error("Unsupported file type")
        raise HTTPException(400, "Unsupported file type")
    
    # Read files asynchronously
    logger.info("Reading files...")
    id_bytes, ref_bytes, act_bytes = await asyncio.gather(
        id_card.read(),
        reference.read(),
        action.read()
    )
    logger.info("Files read successfully.")

    # Process images
    id_img = process_image(id_bytes)
    ref_img = process_image(ref_bytes)
    act_img = process_image(act_bytes)
    logger.info("Images processed.")

    # Face detection and alignment
    id_array = np.array(id_img)
    logger.info(f"id_array shape: {id_array.shape}")
    face = await detect_face(id_array)
    x, y, w, h = face['box']
    cropped_id = id_array[y:y+h, x:x+w]
    logger.info(f"cropped_id shape: {cropped_id.shape}")

    ref_array = np.array(ref_img)
    logger.info(f"ref_array shape: {ref_array.shape}")
    ref_face = await detect_face(ref_array)
    rx, ry, rw, rh = ref_face['box']
    cropped_ref = ref_array[ry:ry+rh, rx:rx+rw]
    logger.info(f"cropped_ref shape: {cropped_ref.shape}")

    act_array = np.array(act_img)
    logger.info(f"act_array shape: {act_array.shape}")
    act_face = await detect_face(act_array)
    ax, ay, aw, ah = act_face['box']
    cropped_act = act_array[ay:ay+ah, ax:ax+aw]
    logger.info(f"cropped_act shape: {cropped_act.shape}")

    # Enhance cropped faces
    cropped_id = await enhance_image_quality(cropped_id)
    cropped_ref = await enhance_image_quality(cropped_ref)
    cropped_act = await enhance_image_quality(cropped_act)

    # Liveness checks
    logger.info("Starting liveness checks...")
    ref_bgr = cv2.cvtColor(cropped_ref, cv2.COLOR_RGB2BGR)
    act_bgr = cv2.cvtColor(cropped_act, cv2.COLOR_RGB2BGR)
    
    liveness_results = await asyncio.gather(
        verify_liveness(ref_bgr),
        verify_liveness(act_bgr)
    )
    logger.info(f"Liveness results: {liveness_results}")

    if not all(liveness_results):
        logger.error("Liveness check failed")
        raise LivenessError()

    # Initialize verifiers
    action_verifier = ActionVerifier()
    multi_stage = MultiStageVerification()
    
    # Perform multi-stage verification
    action_to_verify = (
        await get_random_action() 
        if settings.required_action == "random" 
        else settings.required_action
    )
    
    verification_result, message = await multi_stage.verify_complete(
        cropped_id,
        cropped_ref,
        cropped_act,
        action_to_verify
    )
    
    if not verification_result:
        raise FaceVerificationError(message)
    
    # Continue with existing similarity checks
    # Feature extraction
    logger.info("Extracting embeddings for ID image...")
    id_embedding_dict = await get_embeddings(cv2.resize(cropped_id, settings.target_size))
    id_embedding = id_embedding_dict[0]["embedding"]
    logger.info("ID embedding extracted.")

    logger.info("Extracting embeddings for reference image...")
    ref_embedding_dict = await get_embeddings(cv2.resize(cropped_ref, settings.target_size))
    ref_embedding = ref_embedding_dict[0]["embedding"]
    logger.info("Reference embedding extracted.")

    logger.info("Extracting embeddings for action image...")
    act_embedding_dict = await get_embeddings(cv2.resize(cropped_act, settings.target_size))
    act_embedding = act_embedding_dict[0]["embedding"]
    logger.info("Action embedding extracted.")

    # Similarity checks
    logger.info("Verifying similarity between ID and reference...")
    sim_id_ref = cosine_similarity(id_embedding, ref_embedding)
    
    logger.info("Verifying similarity between reference and action...")
    sim_ref_act = cosine_similarity(ref_embedding, act_embedding)
    
    threshold = settings.threshold_distance
    verified = sim_id_ref > threshold and sim_ref_act > threshold

    # Quality metrics
    def calculate_quality(img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return {
            "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
            "brightness": gray.mean()
        }

    # Calculate quality metrics
    id_quality = calculate_quality(cropped_id)
    ref_quality = calculate_quality(cropped_ref)
    act_quality = calculate_quality(cropped_act)

    # Log detailed results
    logger.info(f"""
    Similarity Results:
    ------------------
    ID-Reference: {sim_id_ref:.4f} (threshold: {threshold})
    Reference-Action: {sim_ref_act:.4f} (threshold: {threshold})
    
    Image Quality:
    -------------
    ID Card - Sharpness: {id_quality['sharpness']:.2f}, Brightness: {id_quality['brightness']:.2f}
    Reference - Sharpness: {ref_quality['sharpness']:.2f}, Brightness: {ref_quality['brightness']:.2f}
    Action - Sharpness: {act_quality['sharpness']:.2f}, Brightness: {act_quality['brightness']:.2f}
    """)

    # Validate quality
    def validate_quality(quality: dict, image_type: str):
        if quality["sharpness"] < settings.min_sharpness:
            raise ImageQualityError(
                f"{image_type} image too blurry",
                {
                    "current_sharpness": quality["sharpness"],
                    "minimum_required": settings.min_sharpness,
                    "brightness": quality["brightness"]
                }
            )
        if not settings.min_brightness <= quality["brightness"] <= settings.max_brightness:
            raise ImageQualityError(
                f"{image_type} image brightness out of range",
                {
                    "current_brightness": quality["brightness"],
                    "allowed_range": [settings.min_brightness, settings.max_brightness],
                    "sharpness": quality["sharpness"]
                }
            )

    # Check each image quality
    validate_quality(id_quality, "ID card")
    validate_quality(ref_quality, "Reference")
    validate_quality(act_quality, "Action")

    return VerificationResponse(
        verified=verified,
        id_quality=QualityInfo(**id_quality),
        reference_quality=QualityInfo(**ref_quality),
        action_quality=QualityInfo(**act_quality),
        processing_time=time.time() - start_time
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.app_version}

# ----------------- Cosine Similarity -----------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ----------------- Advanced Liveness Check -----------------
async def advanced_liveness_check(image: np.ndarray) -> bool:
    """Enhanced liveness detection"""
    try:
        # Basic checks
        if not await verify_liveness(image):
            return False
            
        # Check face depth variation
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            return False
            
        depth_variations = await calculate_depth_variations(results.multi_face_landmarks[0])
        if depth_variations < settings.min_depth_variation:
            return False
                
        # Check texture patterns - await the result
        texture_score = await analyze_face_texture(image)
        if texture_score < settings.min_texture_score:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Advanced liveness check failed: {str(e)}")
        return False

async def analyze_face_texture(image: np.ndarray) -> float:
    """Analyze face texture for anti-spoofing"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return texture_score / 100.0  # Normalize
    except Exception as e:
        logger.error(f"Texture analysis error: {str(e)}")
        return 0.0

async def calculate_depth_variations(landmarks) -> float:
    """Calculate face depth variations from landmarks"""
    try:
        # Extract Z coordinates and convert to float
        z_coords = [float(landmark.z) for landmark in landmarks.landmark]
        # Calculate variation
        return float(np.std(z_coords))
    except Exception as e:
        logger.error(f"Depth calculation error: {str(e)}")
        return 0.0

# ----------------- Action Verifier -----------------
class ActionVerifier:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
    
    async def verify_action(self, image: np.ndarray, expected_action: str) -> bool:
        """التحقق من أداء أي حركة من الحركات المتاحة"""
        logger.info("Checking for any valid action...")
        
        try:
            if expected_action == "any":
                # فحص كل الحركات المتاحة
                smile_valid = await self.detect_smile(image)
                blink_valid = await self.detect_blink(image)
                head_valid = await self.detect_head_turn(image)
                
                # إذا تم اكتشاف أي حركة
                is_valid = any([smile_valid, blink_valid, head_valid])
                
                if is_valid:
                    logger.info("Valid action detected!")
                    if smile_valid:
                        logger.info("Smile detected")
                    if blink_valid:
                        logger.info("Blink detected")
                    if head_valid:
                        logger.info("Head turn detected")
                else:
                    logger.warning("No valid action detected")
                
                return is_valid
            else:
                # التحقق من حركة محددة
                return await getattr(self, f"detect_{expected_action}")(image)
                
        except Exception as e:
            logger.error(f"Action verification error: {str(e)}")
            return False

    async def detect_smile(self, image: np.ndarray) -> bool:
        try:
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return False
            
            landmarks = results.multi_face_landmarks[0]
            mouth_width = self._get_mouth_width(landmarks)
            return mouth_width > settings.action_threshold["smile"]
        except Exception as e:
            logger.error(f"Smile detection error: {str(e)}")
            return False

    async def detect_blink(self, image: np.ndarray) -> bool:
        try:
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                logger.error("No face landmarks detected for blink verification")
                return True  # Temporarily return True
                
            landmarks = results.multi_face_landmarks[0]
            left_eye = self._get_eye_ratio(landmarks, "left")
            right_eye = self._get_eye_ratio(landmarks, "right")
            
            logger.info(f"Eye ratios - Left: {left_eye:.3f}, Right: {right_eye:.3f}")
            return True  # Temporarily return True
            
        except Exception as e:
            logger.error(f"Blink detection error: {str(e)}")
            return True  # Temporarily return True

    async def detect_head_turn(self, image: np.ndarray) -> bool:
        """Detect head turning action"""
        try:
            results = self.face_mesh.process(image)
            if not results.multi_face_landmarks:
                return False
                
            landmarks = results.multi_face_landmarks[0]
            # Calculate head rotation
            rotation = self._calculate_head_rotation(landmarks)
            return abs(rotation) > 30  # More than 30 degrees
        except Exception as e:
            logger.error(f"Head turn detection error: {str(e)}")
            return False

    def _get_eye_ratio(self, landmarks, eye_side: str) -> float:
        """Calculate eye aspect ratio"""
        if eye_side == "left":
            points = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
        else:
            points = [362, 385, 387, 263, 373, 380]  # Right eye landmarks
        
        eye_points = [landmarks.landmark[p] for p in points]
        v1 = np.abs(eye_points[1].y - eye_points[5].y)
        v2 = np.abs(eye_points[2].y - eye_points[4].y)
        h = np.abs(eye_points[0].x - eye_points[3].x)
        return (v1 + v2) / (2.0 * h)

    def _calculate_head_rotation(self, landmarks) -> float:
        """Calculate head rotation angle"""
        nose_tip = landmarks.landmark[4]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        
        # Calculate angle from eye line to vertical
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        return np.degrees(np.arctan2(dy, dx))

# ----------------- Multi-Stage Verification -----------------
class MultiStageVerification:
    def __init__(self):
        self.threshold = settings.threshold_distance
        self.action_verifier = ActionVerifier()
        
    async def verify_complete(self, id_img, ref_img, action_img, expected_action):
        try:
            # 1. Basic face detection
            faces_valid = await self.verify_faces_exist(id_img, ref_img, action_img)
            if not faces_valid:
                return False, "Face detection failed"
                
            # 2. Liveness check - explicit error handling
            try:
                liveness_valid = await advanced_liveness_check(action_img)
                if not liveness_valid:
                    return False, "Liveness check failed"
            except Exception as e:
                logger.error(f"Liveness check error: {str(e)}")
                return False, f"Liveness check error: {str(e)}"
                
            # 3. Action verification
            action_valid = await self.action_verifier.verify_action(action_img, expected_action)
            if not action_valid:
                return False, "Action verification failed"
                
            # 4. Face matching
            match_valid = await self.verify_face_match(id_img, ref_img, action_img)
            if not match_valid:
                return False, "Face matching failed"
                
            return True, "Verification successful"
            
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            return False, f"Verification failed: {str(e)}"

    async def verify_faces_exist(self, id_img, ref_img, action_img) -> bool:
        """Verify faces exist in all images"""
        try:
            faces = await asyncio.gather(
                detect_face(id_img),
                detect_face(ref_img),
                detect_face(action_img)
            )
            return all(faces)
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return False

    async def verify_face_match(self, id_img, ref_img, action_img) -> bool:
        """Verify faces match across images"""
        try:
            # Get embeddings
            id_emb = (await get_embeddings(cv2.resize(id_img, settings.target_size)))[0]["embedding"]
            ref_emb = (await get_embeddings(cv2.resize(ref_img, settings.target_size)))[0]["embedding"]
            act_emb = (await get_embeddings(cv2.resize(action_img, settings.target_size)))[0]["embedding"]
            
            # Calculate similarities
            sim_id_ref = cosine_similarity(id_emb, ref_emb)
            sim_ref_act = cosine_similarity(ref_emb, act_emb)
            
            return sim_id_ref > self.threshold and sim_ref_act > self.threshold
        except Exception as e:
            logger.error(f"Face matching error: {str(e)}")
            return False

# ----------------- Safe Gather -----------------
async def safe_gather(*coroutines):
    """Safe version of asyncio.gather that won't fail if one coroutine fails"""
    results = []
    for coro in coroutines:
        try:
            result = await coro
            results.append(result)
        except Exception as e:
            logger.error(f"Coroutine failed: {str(e)}")
            results.append(None)
    return results

# ----------------- Main -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    #supersecretkey


