# Adversarial Attack Challenge

2025 Adversarial Attack Challenge for Secure Face Recognition. This project is designed to advance research in adversarial robustness and real-world security applications by testing models in two distinct tracks.

For full details about challenge, visit organizers website https://www.youverse.id/adversarial.

## Detection Track 
Identify whether a face image is adversarial or clean.


## Resilience Track

Train a robust face recognition model that maintains high accuracy even under adversarial attacks.

**Adversarial Attacks Overview**:

- **Impersonation Attacks**: These attacks attempt to fool the face recognition system into incorrectly recognizing an attacker as a specific target individual. For example, modifying an input image so that the model falsely identifies the attacker as someone else.
- **Evasion Attacks**: These attacks aim to prevent the system from recognizing the attacker at all. The adversary modifies their appearance or the input in a way that causes the model to fail in identifying them or to classify them as an unknown person.

The goal is to develop models that are resilient to both types of attacks while maintaining high recognition performance on clean (non-adversarial) data.

## 📦 Getting the Data

To extract the dataset from the `.7z` archive, make sure you have `p7zip` installed.

### 🔧 Install p7zip (if not already installed)

```bash
brew install p7zip
```

### 📂 Extract the Archive

Run the following command to extract the contents of Challenge.7z:

```bash
7z x Challenge.7z
```

### 🔑 Archive Password

The archive is password-protected. Use the following password when prompted:

```
6V3t0aDlFK5wPZm
```

## 🗂️ Dataset

final_attack_attackid label files columns

- attacked_image: The result of an adversarial attack (the image that has been altered).
- original_identity: The identity of the person in the attacked_image before the attack.
- attack_type: Type of attack. Impersonation or evasion.
- target_image: An image of the person the attacker is trying to impersonate. In case of evasion attack to avoid being recognized as anyone, especially not as themselves.
- attack_id: The ID of the attack attempt.