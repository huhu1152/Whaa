// قائمة الفئات المحدثة بناءً على الملفات التي قدمتها
const classNames = [
    "حجر الفمر",
    "حجر الهياب",
    "حجر الباقوت",
    "حجر البشب",
    "حجر البشم",
    "حجر فيبروز",
    "در النجف",
    "عباس ابلد",
    "عقيق",
    "عقيق - شرك الشمس",
    "الواقع الأبيض",
    "الواقع الأسود",
    "الزمرد",
    "حجر الأزورد",
    "حجر الأوبال",
    "حجر الجمشت",
    "حجر الحديد الصيني (الهيمتايت)",
    "حجر الزفير",
    "حجر السلطاني",
    "حجر القمر"
];

let model;
let uploadedImage;

// عناصر DOM
const uploadArea = document.getElementById('uploadArea');
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const confidenceDiv = document.getElementById('confidence');
const loadingDiv = document.getElementById('loading');
const modelStatus = document.getElementById('modelStatus');

// تحميل النموذج عند فتح الصفحة
document.addEventListener('DOMContentLoaded', async function() {
    await loadModel();
});

async function loadModel() {
    try {
        modelStatus.innerHTML = '<i class="fas fa-sync fa-spin"></i> جاري تحميل نموذج الذكاء الاصطناعي...';
        
        // المسار المعدل لملف النموذج على GitHub
        model = await tf.loadGraphModel('https://raw.githubusercontent.com/[اسم-المستخدم]/[اسم-المستودع]/main/model/model.json');
        
        modelStatus.innerHTML = '<i class="fas fa-check-circle"></i> النموذج جاهز للاستخدام!';
        console.log('تم تحميل النموذج بنجاح');
    } catch (error) {
        console.error('حدث خطأ أثناء تحميل النموذج:', error);
        modelStatus.innerHTML = '<i class="fas fa-exclamation-circle"></i> فشل في تحميل النموذج. تأكد من صحة المسارات.';
    }
}

// معالجة تحميل الصورة
uploadArea.addEventListener('click', () => imageUpload.click());

imageUpload.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadedImage = new Image();
            uploadedImage.src = e.target.result;
            uploadedImage.onload = function() {
                imagePreview.src = this.src;
                imagePreview.style.display = 'block';
                predictBtn.disabled = false;
            };
        };
        reader.readAsDataURL(file);
    }
});

// التنبؤ بنوع الحجر
async function predictImage() {
    if (!uploadedImage) {
        alert('الرجاء تحميل صورة أولاً');
        return;
    }
    
    if (!model) {
        alert('جاري تحميل النموذج، الرجاء الانتظار...');
        await loadModel();
        if (!model) {
            alert('لا يمكن تحميل النموذج');
            return;
        }
    }
    
    try {
        // إظهار مؤشر التحميل
        loadingDiv.style.display = 'block';
        resultDiv.textContent = '';
        confidenceDiv.textContent = '';
        
        // معالجة الصورة
        const tensor = tf.browser.fromPixels(uploadedImage)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims();
        
        // التنبؤ (مع إضافة تأخير لمحاكاة المعالجة)
        setTimeout(async () => {
            const predictions = await model.predict(tensor).data();
            
            // إيجاد أعلى تنبؤ
            let maxPrediction = 0;
            let predictedClassIndex = 0;
            
            for (let i = 0; i < predictions.length; i++) {
                if (predictions[i] > maxPrediction) {
                    maxPrediction = predictions[i];
                    predictedClassIndex = i;
                }
            }
            
            const predictedClassName = classNames[predictedClassIndex];
            const confidence = (maxPrediction * 100).toFixed(2);
            
            // عرض النتيجة
            resultDiv.textContent = `نوع الحجر: ${predictedClassName}`;
            confidenceDiv.textContent = `مستوى الثقة: ${confidence}%`;
            
            // إخفاء مؤشر التحميل
            loadingDiv.style.display = 'none';
            
            // تحرير الذاكرة
            tensor.dispose();
        }, 1500);
    } catch (error) {
        console.error('حدث خطأ أثناء التنبؤ:', error);
        resultDiv.textContent = 'حدث خطأ أثناء معالجة الصورة';
        confidenceDiv.textContent = '';
        loadingDiv.style.display = 'none';
    }
}

// تعيين حدث للزر
predictBtn.addEventListener('click', predictImage);
